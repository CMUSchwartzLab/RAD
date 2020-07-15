""" generate simulated data

"""

import numpy as np


__author__ = "Yifeng Tao"


def gen_F(n_comp=3, n_samp=100):
  """ generate fraction matrix F
  
  """

  # https://math.stackexchange.com/questions/502583/uniform-sampling-of-points-on-a-simplex
  F = np.random.uniform(size=(n_comp,n_samp))
  F = -np.log(F)
  F_sum = np.sum(F, axis=0)
  F = F/F_sum
  
  return F


def gen_C(n_gene, n_comp, n_path, sig):
  """ generate clean component matrix C (noise not added yet)
  
  """
  
  gpg = n_gene // n_path

  mu = np.array([0]*gpg)
  r = 0.95*np.ones((gpg,gpg))+np.diag([0.05]*gpg)
  
  #r = np.array([
  #        [  1, 0.95, 0.95, 0.95, 0.95],
  #        [ 0.95,  1,  0.95, 0.95, 0.95],
  #        [ 0.95,  0.95,  1, 0.95, 0.95],
  #        [ 0.95,  0.95, 0.95,  1, 0.95],
  #        [ 0.95,  0.95, 0.95, 0.95,  1]
  #    ])
  
  C = [sig*np.random.multivariate_normal(mu, r, size=3).T+6.0 for _ in range(n_path)]
  C = np.vstack(C)

  C = 2**C

  return C


def gen_noise(n_gene, n_samp, f=0.1):
  """ generate noise matrix
  
  """
  
  noise = np.random.normal(size=(n_gene, n_samp))*f

  noise = 2**noise

  sgn = 2*np.random.randint(0,2,(n_gene, n_samp))-1

  noise = sgn*noise

  return noise


def gen_sim_data(n_samp=100, n_comp=3, n_gene=2048, n_modu=64, f=4.0):
  """ generate simulated data
  
  Parameters
  ----------
  n_samp: int
    number of bulk samples
  n_comp: int
    number of cell components/populations
  n_gene: int
    number of genes to be considered
  n_modu: int
    number of gene modules, the number of genes per module is n_gene//n_modu
  f: float
    noise level
    
  Returns
  -------
  Cgt: 2D array of float
    ground truth C without noise
  Fgt: 2D array of float
    ground truth F withou noise
  B: 2D array of float
    noisy given bulk data
  module: list of list of int
    each sublist contains indices of genes in the same gene module

  """
  
  gpg = int(n_gene/n_modu)
  n_gene = n_modu*gpg
  
  sig = 2.5
  
  # ground truth C and F
  Cgt = gen_C(n_gene, n_comp, n_modu, sig)
  Fgt = gen_F(n_comp=n_comp, n_samp=n_samp)
  
  noise = gen_noise(n_gene, n_samp, f=f)
  
  # noisy bulk data B
  B = np.clip(np.dot(Cgt, Fgt) + noise, 0, None)
  
  module = [
      [idx_m*gpg+idx_g for idx_g in range(gpg)] 
      for idx_m in range(n_modu)]
    
  return Cgt, Fgt, B, module


