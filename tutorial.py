""" Tutorial of RAD toolkit

  introduces how to utilize the main APIs through a simulated dataset:
    compress_module
    estimate_number
    estimate_clones
    estimate_marker

"""

import numpy as np

from itertools import permutations
from sklearn.metrics import r2_score

from gen_sim_data import gen_sim_data

from utils import get_min_ssd_index

from rad import compress_module, estimate_number, estimate_clones, estimate_marker

import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Yifeng Tao"



sns.set_style("white")

def plot(C, F, Cgt, Fgt, plotfig=True):

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

    plt.plot([0,1],[0,1],'--', label='$R_F^2$=%.3f, MSE=%.3f'%(r2f, mse),color='gray')

    plt.scatter(Fgt[0],Fp[0], label='Population 1')
    plt.scatter(Fgt[1],Fp[1], label='Population 2')
    plt.scatter(Fgt[2],Fp[2], label='Population 3')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    #plt.legend(fancybox=True, framealpha=0.3)
    plt.legend(frameon=False)

    plt.xlabel('Ground truth abundance ($F$)')
    plt.ylabel('Estimated abundance ($\hat{F}$)')

    ax = plt.subplot(1,2,2)

    plt.plot([0, 15], [0, 15], '--', color='gray',label=r'$R_C^2$=%.3f, $L_1$ loss=%.3f'%(r2c,l1_loss))

    plt.scatter(x, y, s=1, edgecolor='', alpha=0.2)

    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.legend(frameon=False)
    plt.xlabel('Ground truth expression ($\log_2 C$)')
    plt.ylabel('Estimated expression ($\log_2 \hat{C}$)')
    
  return r2f, mse, r2c, l1_loss



# We first generate a simulated dataset B \in R_{+}^{2048 \times 100} for analysis,
# where we have 100 bulk samples and 2048 genes.
# Each bulk sample is a combination of 3 cell populations.
# Each 32 genes belongs to a module and coexpress, which lead to 64 modules.
# The bulk sample also contains noise.

# We know the ground truth Cgt and Fgt, the gene module knowledge.
# Our aim is to recover the component matrix C, fraction matrix F from them, 
# such that C * F ~= B
  
Cgt, Fgt, B, module = gen_sim_data()

# We first compress the bulk gene expression into bulk module expression
B_M = compress_module(B, module)


# Notice that we do not know the number of components k,
# Therefore, we will estimate it.
# This step can take a long time, here use max_comp=4 and n_splits=2 to reduce time
k = estimate_number(B_M, max_comp=10, n_splits=20)


# Now given both compressed bulk data B_M and estimated number of cell populations k,
# we utilize the core RAD agorithm to unmix cell component C_M and fractions F.
C_M, F = estimate_clones(B_M, k, verbose=False)


# Finally, if we have the bulk data of other biomarkers B_P, 
# we can estimate the C_P from B_P and F.
# But here we are interested in the biomarker of original gene expressions B.
C = estimate_marker(B, F)

# We evaluate the estimation accuray
r2f, mse, r2c, l1 = plot(C, F, Cgt, Fgt, plotfig=True)

print('r2f=%.3f, mse=%.3f'%(r2f, mse))

