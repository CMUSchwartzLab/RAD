""" utils functions to support RAD and analysis

"""

import numpy as np

from itertools import permutations
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Yifeng Tao"


def mask_fro_norm(B, M, C, F):
  """Compute the beta-divergence of B and dot(C, F).
  
  Parameters
  ----------
  B : float or array-like, shape (n_samples, n_features)
  C : float or dense array-like, shape (n_samples, n_components)
  F : float or dense array-like, shape (n_components, n_features)
  beta : float, string in {"frobenius", "kullback-leibler", "itakura-saito"}
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
  
  min_ssd = float("inf")
  min_indexCgt = None
  for indexCgt in IndexCgt:
    sum_square_distance = get_sum_square_distance(C, Cgt, indexCgt)
    if sum_square_distance < min_ssd:
      min_indexCgt = indexCgt
      min_ssd = sum_square_distance
      
  return min_indexCgt, min_ssd


def evaluate_accuracy(C, F, Cgt, Fgt, plotfig=True):
  """ evaluate the performance of deconvolution estimation
  
  Parameters
  ----------
  C: 2D array of float
    estimated component matrix
  F: 2D array of float
    estimated fraction matrix
  Cgt: 2D array of float
    ground truth component matrix
  Fgt: 2D array of float
    ground truth component matrix
  plotfig: boolen
    whether to plot figure or not
      
  Returns
  -------
  r2f: float
    R^2 value of flattened F and Fgt
  mse: float
    mean square error of flattened F and Fgt
  r2c: float
    R^2 value of flattened C and Cgt
  l1_loss: float
    L1 loss of flattend C and Cgt
  """
  
  sns.set_style("white")

  IndexCgt = list(permutations(range(0, 3)))

  min_indexCgt, min_ssd = get_min_ssd_index(C, Cgt, IndexCgt)

  # aligned predicted C and F matrices
  Cp = np.zeros(C.shape)
  Fp = np.zeros(F.shape)
  for i, j in enumerate(min_indexCgt):
    Cp[:,j] = C[:,i]
    Fp[j,:] = F[i,:]

  r2f = r2_score(Fgt.reshape(-1),Fp.reshape(-1))
  d = Fgt.reshape(-1)-Fp.reshape(-1)
  mse = np.dot(d,d)/len(d)

  l1_loss = np.sum(np.abs(Cgt.reshape(-1) - Cp.reshape(-1)))/np.sum(Cgt.reshape(-1))

  r2c = r2_score(Cgt.reshape(-1),Cp.reshape(-1))

  x, y = np.log2(Cgt.reshape(-1)+1),np.log2(Cp.reshape(-1)+1)

  if plotfig:

    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(1,2,1)

    plt.plot([0,1],[0,1],"--", label="$R_F^2$=%.3f, MSE=%.3f"%(r2f, mse),color="gray")

    plt.scatter(Fgt[0],Fp[0], label="Population 1")
    plt.scatter(Fgt[1],Fp[1], label="Population 2")
    plt.scatter(Fgt[2],Fp[2], label="Population 3")

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    plt.legend(frameon=False)

    plt.xlabel("Ground truth abundance ($F$)")
    plt.ylabel("Estimated abundance ($\hat{F}$)")

    ax = plt.subplot(1,2,2)

    plt.plot([0, 15], [0, 15], "--", color="gray",label=r"$R_C^2$=%.3f, $L_1$ loss=%.3f"%(r2c,l1_loss))

    plt.scatter(x, y, s=1, edgecolor="", alpha=0.2)

    # hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    plt.legend(frameon=False)
    plt.xlabel("Ground truth expression ($\log_2 C$)")
    plt.ylabel("Estimated expression ($\log_2 \hat{C}$)")
    
  return r2f, mse, r2c, l1_loss