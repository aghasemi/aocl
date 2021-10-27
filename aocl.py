import sklearn as skl
import sklearn.datasets as skd
import numpy as np
from sklearn.neighbors import KernelDensity

def score_base(u, X, kernel): # Equation 7 in the paper
  if len(np.shape(u))!=1:  raise ValueError(f'Input vector {u} must have exactly one row')
  n = len(X)
  X_plus = np.vstack((X,u))
  score = 0
  for i in range(n+1):
    x_i = X_plus[i,:]
    X_minus_i = np.delete(X_plus, (i), axis=0)
    f_hat = kernel.fit(X_minus_i)
    score_i = f_hat.score_samples([x_i])
    score += score_i
  return score/(n+1)

def score_base_with_outliers(u, X,Y, kernel): # Equations 9 and 11
  if len(np.shape(u))!=1:  raise ValueError(f'Input vector {u} must have exactly one row')

  n = len(X)
  m = len(Y) #Y is the set of ouliers
  score_inliers = score_base(u, X, kernel) # The first term in the parantheses
  X_plus = np.vstack((X,u))
  f_hat = kernel.fit(X_plus)
  score_outliers = 0
  for j in range(m):
    y_j = Y[j,:]
    score_j = f_hat.score_samples([y_j])
    score_outliers += score_j
  score_outliers = score_outliers / (m)
  return ( score_inliers - score_outliers )

def score_base_with_outliers_and_prior(u, X,Y, kernel): # Equation 12
  if len(np.shape(u))!=1:  raise ValueError(f'Input vector {u} must have exactly one row')

  n = len(X)
  m = len(Y) #Y is the set of ouliers
  score_inliers = 0 # The first term in the parantheses is independent of u, so we actually do not need to compute it!
  
  f_hat = kernel.fit(X)
  score_outliers = f_hat.score_samples([u]) / (m+1) # Further simplifying also the second term in Eq. 12 and keeping only the terms that depend on u
  
  return ( score_inliers - score_outliers )


def score_1(u, X, kernel): # Equation 7
  return score_base(u, X, kernel)

def score_2(u, X, kernel): # Equation 8
  current_score = kernel.fit(X).score_samples([u])
  return current_score * score_base(u, X, kernel)

def score_3(u, X,Y, kernel): # Equation 9
  current_score = kernel.fit(X).score_samples([u])
  return current_score * score_base_with_outliers(u, X,Y, kernel)

def score_4(u, X,Y, prior_prob, kernel): # Equation 10
  return prior_prob * score_base_with_outliers(u, X, Y, kernel) + (1-prior_prob) * score_base_with_outliers_and_prior(u, X, Y, kernel)


if __name__=="__main__":
	n_tra = 200
	n_val = 200
	k = 2
	gaussian_kernel = KernelDensity(kernel='gaussian', bandwidth=0.2)

	X, y  = skl.datasets.make_blobs(n_samples=[n_tra+n_val,100,50,20],n_features=k)

	inl_ind = np.argwhere(y==0).flatten()
	outl_ind = np.argwhere(y>=2).flatten()

	X_tra = X[inl_ind[:n_tra],:] # Inlier training data
	X_val = X[inl_ind[n_tra:],:] # Inlier validation data
	X_val_outl = X[outl_ind,:] # Outlier validation data

	print(score_1(X_val[9,:], X_tra, gaussian_kernel))
	print(score_2(X_val[9,:], X_tra, gaussian_kernel))
	print(score_3(X_val[9,:], X_tra, X_val_outl, gaussian_kernel))
	print(score_4(X_val[9,:], X_tra, X_val_outl, .9, gaussian_kernel))