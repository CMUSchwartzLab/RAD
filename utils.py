""" utils functions to support RAD and analysis

"""

import numpy as np

__author__ = "Yifeng Tao"


def mask_fro_norm(B, M, C, F):
  """Compute the beta-divergence of B and dot(C, F).
  
  Parameters
  ----------
  B : float or array-like, shape (n_samples, n_features)
  C : float or dense array-like, shape (n_samples, n_components)
  F : float or dense array-like, shape (n_components, n_features)
  beta : float, string in {'frobenius', 'kullback-leibler', 'itakura-saito'}
      Parameter of the beta-divergence.
      If beta == 2, this is half the Frobenius *squared* norm.
  square_root : boolean, default False
      If True, return np.sqrt(2 * res)
      For beta == 2, it corresponds to the Frobenius norm.
      
  Returns
  -------
      res : float
          Beta divergence of B and np.dot(C, F)
  """

  Y = np.absolute( np.multiply(M, B-np.dot(C, F)) )

  res = np.sqrt( np.sum(np.multiply(Y, Y)) )

  return res


def mask_mse(B, M, C, F):

  Y = np.absolute( np.multiply(M, B-np.dot(C, F)) )

  res = 1.0*np.sum(np.multiply(Y, Y))/np.sum(M)

  return res


def get_square_distance(v1, v2):
  
  v = v1 - v2
  square_distance = np.dot(v, v)
  
  return square_distance


def get_sum_square_distance(C, Cgt, indexCgt):
  
  sum_square_distance = 0
  for i, j in enumerate(indexCgt):
    v1 = C[:,i]
    v2 = Cgt[:,j]
    square_distance = get_square_distance(v1, v2)
    sum_square_distance += square_distance
    
  return 1.0*sum_square_distance/len(indexCgt)


def get_min_ssd_index(C, Cgt, IndexCgt):
  
  min_ssd = float('inf')
  min_indexCgt = None
  for indexCgt in IndexCgt:
    sum_square_distance = get_sum_square_distance(C, Cgt, indexCgt)
    if sum_square_distance < min_ssd:
      min_indexCgt = indexCgt
      min_ssd = sum_square_distance
      
  return min_indexCgt, min_ssd

