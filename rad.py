""" Robust and Accurate Deconvolution (RAD) toolkit

  main APIs:
    compress_module
    estimate_number
    estimate_clones
    estimate_marker

"""

import random
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import cvxopt

from utils import mask_fro_norm, mask_mse

import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Yifeng Tao"


def compress_module(B, module):
  """ compress raw gene expression into module expression
  
  Parameters
  ----------
  B: 2D array of non-negative float
    bulk gene data, each row a sample, each column a gene
  module: list of list of int
    each sublist contains indices of genes in the same gene module
    
  Returns
  -------
  B_M: 2D array of float
    compressed module-level bulk data
  
  """
  
  B_M = np.array([np.mean([B[idx] for idx in m],axis=0) for m in module])
  
  return B_M


# the multiplicative update phase borrows code from scikit-learn:
# https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/decomposition/nmf.py#L699

def _normalize_frac(F):
  col_sums = F.sum(axis=0)
  return F / col_sums[np.newaxis, :]


def _initialize_nmf(B, n_components, init=None, eps=1e-6,
                    random_state=None):
    """ Algorithms for NMF initialization.
    Computes an initial guess for the non-negative
    rank k matrix approximation for B: B = CF
    
    Parameters
    ----------
    B : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: 'nndsvd' if n_components < n_features, otherwise 'random'.
        Valid options:
        - 'random': non-negative random matrices, scaled with:
            sqrt(B.mean() / n_components)
    eps : float
        Truncate all values less then this in output to zero.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``random`` == 'nndsvdar' or 'random'.
        
    Returns
    -------
    C : array-like, shape (n_samples, n_components)
        Initial guesses for solving B ~= CF
    F : array-like, shape (n_components, n_features)
        Initial guesses for solving B ~= CF
        
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    
    """

    n_samples, n_features = B.shape

    # Random initialization
    avg = np.sqrt(B.mean() / n_components)
    rng = np.random.mtrand._rand

    F = avg * rng.randn(n_components, n_features)
    C = avg * rng.randn(n_samples, n_components)
    # we do not write np.abs(F, out=F) to stay compatible with
    # numpy 1.5 and earlier where the 'out' keyword is not
    # supported as a kwarg on ufuncs

    np.abs(C, C)
    np.abs(F, F)
    F = _normalize_frac(F)

    return C, F


def _multiplicative_update_f_mask(B, M, C, F):
  """ update F in Multiplicative Update NMF
  
  """
  
  numerator = np.dot(C.T, np.multiply(M, B))
  denominator = np.dot( C.T, np.multiply(M,np.dot(C,F)) )
  numerator /= denominator
  delta_F = numerator

  return delta_F


def _multiplicative_update_c_mask(B, M, C, F):
  """ update C in Multiplicative Update NMF
  
  """

  numerator = np.dot(np.multiply(M,B), F.T)
  denominator = np.dot( np.multiply(M,np.dot(C,F)), F.T )

  numerator /= denominator
  delta_C = numerator

  return delta_C


def _fit_multiplicative_update_mask(B, M, C, F, max_iter=200, tol=1e-4, verbose=True):
    """ Compute Non-negative Matrix Factorization with Multiplicative Update
    The objective function is mask_fro_norm(B, CF) and is minimized with an
    alternating minimization of C and F. Each minimization is done with a
    Multiplicative Update.
    
    Parameters
    ----------
    B : array-like, shape (n_samples, n_features)
        Constant input matrix.
    C : array-like, shape (n_samples, n_components)
        Initial guess for the solution.
    F : array-like, shape (n_components, n_features)
    
    Returns
    -------
    C : array, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.
    F : array, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    n_iter : int
        The number of iterations done by the algorithm.
        
    """

    start_time = time.time()

    error_at_init = mask_fro_norm(B, M, C, F)
    previous_error = error_at_init

    if verbose:
      iter_time = time.time()
      print("Epoch %02d reached after %.3f seconds, error: %f" %
                        (0, iter_time - start_time, error_at_init))

    for n_iter in range(1, max_iter + 1):
      
        # update C
        delta_C = _multiplicative_update_c_mask(B, M, C, F)
        C *= delta_C

        # update F
        delta_F = _multiplicative_update_f_mask(B, M, C, F)
        F *= delta_F

        # normalize each column of F
        F = _normalize_frac(F)

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = mask_fro_norm(B, M, C, F)
            if verbose:
                iter_time = time.time()
                print("Epoch %02d reached after %.3f seconds, error: %f" %
                      (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 200 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    return C, F, n_iter, previous_error


def _cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def _quad_prog_BF2C(B, F, M, max_val=2**19):
  """ Solve the QP problem of
  
    Minimize_C ||B - C F||_2^2
    Subject to C >= 0
               C <= max_val
               
  """

  num_gene = B.shape[0]
  num_comp = F.shape[0]

  Gl = np.diag([-1.0]*num_comp)
  hl = np.zeros(num_comp).reshape((num_comp,))

  Gu = np.diag([1.0]*num_comp)
  hu = np.array([max_val]*num_comp).reshape((num_comp,))

  G=np.vstack([Gl, Gu])
  h=np.hstack([hl, hu])

  C = []

  for i in range(num_gene):
    m = M[i,:]
    Fm = [F[:,j] for j, v in enumerate(m) if v != 0]
    Fm = np.array(Fm).T
    P = np.dot(Fm, Fm.T)
    bm = np.array([B[i,j] for j, v in enumerate(m) if v != 0])
    q = -np.dot(Fm, bm)

    ci = _cvxopt_solve_qp(P, q, G, h)

    if ci is None:
      return None

    C.append(ci)

  C = np.vstack(C)

  return C


def estimate_marker(B, F, M=None, max_val=2**19):
  """ Estimate biomarkers of individual components
  
    B_P, F -> C_P or
    B, F -> C
  
  """
  
  if M is None:
    M = np.ones(B.shape)
  
  C = _quad_prog_BF2C(B, F, M, max_val=max_val)
  
  return C
  

def _quad_prog_BC2F(B, C, M):
  """ Solve the QP problem of
  
    Minimize_F ||B - C F||_2^2
    Subject to F >= 0
               \sum_i F_ij =1, for j=1,2,...,num_gene
      
  """

  num_gene = B.shape[0]
  num_smpl = B.shape[1]
  num_comp = C.shape[1]

  Gl = np.diag([-1.0]*num_comp)
  hl = np.zeros(num_comp).reshape((num_comp,))

  Gu = np.diag([1.0]*num_comp)
  hu = np.array([1.0]*num_comp).reshape((num_comp,))

  G=np.vstack([Gl, Gu])
  h=np.hstack([hl, hu])

  A = np.ones(num_comp).reshape((1,num_comp))
  b = np.ones(1).reshape((1,))

  F = []

  for j in range(num_smpl):
    m = M[:,j]

    Cm = [C[i,:] for i, v in enumerate(m) if v != 0]
    Cm = np.array(Cm)

    P = np.dot(Cm.T, Cm)
    bm = np.array([B[i,j] for i, v in enumerate(m) if v != 0])
    q = -np.dot(Cm.T, bm)

    fi = _cvxopt_solve_qp(P, q, G, h, A, b)

    if fi is None:
      return None

    F.append(fi)

  F = np.vstack(F).T

  return F


def _get_sum_InPr(C):
  """ calculate unnormalized cosine similarity and minimize that
  
  """
  
  k = C.shape[1]

  InPr = np.dot(C.T,C)
  sum_InPr = 0
  for i in range(k):
    for j in range(i+1,k):
      sum_InPr += InPr[i,j]

  return sum_InPr


def rad_warmstart(B, M, k, max_iter=2000, tol=1e-4, verbose=True):
  C, F = _initialize_nmf(B, k)
  C, F, n_iter, error = _fit_multiplicative_update_mask(B, M, C, F, max_iter=max_iter, tol=tol, verbose=verbose)

  return C, F, n_iter, error


def rad_coorddescent(B, M, C, F, max_iter=500, tol=1e-4):

  error_at_init = mask_fro_norm(B, M, C, F)

  list_err = [error_at_init]

  previous_error = error_at_init

  for idx_iter in range(max_iter):

    C_tmp = _quad_prog_BF2C(B, F, M)

    if C_tmp is not None:
      C = C_tmp

    error = mask_fro_norm(B, M, C, F)
    list_err.append(error)

    l2 = mask_mse(B, M, C, F)

    F_tmp = _quad_prog_BC2F(B, C, M)
    if F_tmp is not None:
      F = F_tmp
    error = mask_fro_norm(B, M, C, F)
    list_err.append(error)

    l2 = mask_mse(B, M, C, F)

    if (previous_error - error) / error_at_init < tol:
      break
    previous_error = error


  return C, F, list_err


def estimate_clones(B, k, M=None, n_trial=10, verbose=True):
  """ estimate the C and F from B, given k.
  
  the 3rd phase of minimizing unnormalized cosine similarity is included
  
  """
  
  if M is None:
    M = np.ones(B.shape)
    
  min_sum_InPr, min_C, min_F = float('inf'), 0, 0

  for _ in range(n_trial):
    C, F, _, _ = rad_warmstart(B, M, k, verbose=verbose)

    C, F, _ = rad_coorddescent(B, M, C, F)

    sum_InPr = _get_sum_InPr(C)

    if sum_InPr < min_sum_InPr:
      if verbose:
        print(sum_InPr, min_sum_InPr)
      min_C = C
      min_F = F
      min_sum_InPr = sum_InPr

  C, F = min_C, min_F

  return C, F


def estimate_number(B, max_comp=10, n_splits=20, plot_cv_error=True):
  """ Cross-validation of matrix factorization.

  Parameters
  ----------
  B: matrix
    bulk data to be deconvolved.
  n_comp: list int
    numbers of population component.
  n_splits: int
    fold of cross-validation.

  Returns
  -------
  results: dict
    numbers of components, training errors and test errors.
  """
  
  n_comp = [i+1 for i in range(max_comp)]

  results = {
      "n_comp":n_comp,
      "test_error":[[] for _ in range(len(n_comp))],
      "train_error":[[] for _ in range(len(n_comp))]
      }

  rng = [(idx, idy) for idx in range(B.shape[0]) for idy in range(B.shape[1])]
  random.Random(2020).shuffle(rng)

  kf = KFold(n_splits=n_splits)

  idx_fold = 0
  for train_index, test_index in kf.split(rng):
    idx_fold += 1

    rng_train = [rng[i] for i in train_index]
    rng_test = [rng[i] for i in test_index]

    M_test = np.zeros(B.shape)
    for r in rng_test:
      M_test[r[0],r[1]] = 1.0
    M_train = np.zeros(B.shape)
    for r in rng_train:
      M_train[r[0],r[1]] = 1.0

    for idx_trial in range(len(n_comp)):
      dim_k = results["n_comp"][idx_trial]

      C, F = estimate_clones(B, dim_k, M=M_train, verbose=False)

      l2_train = mask_mse(B, M_train, C, F)
      l2_test = mask_mse(B, M_test, C, F)
      results["train_error"][idx_trial].append(l2_train)
      results["test_error"][idx_trial].append(l2_test)

      #print("fold=%3d/%3d, dim_k=%2d, train=%.2e, test=%.2e"%(idx_fold, n_splits, dim_k, l2_train, l2_test))
      
  k = n_comp[np.argmin(np.mean(results['test_error'], axis=1))]
  
  if plot_cv_error:
    plot_cv(results,inputstr="test_error",deno=np.sum(np.multiply(B, B))/B.shape[0]/B.shape[1])

  return k


def plot_cv(results, inputstr="test_error", deno=1.0):
  """ Plot the cross-validation results.

  Parameters
  ----------
  results: dict
  
  """

  size_label = 18
  size_tick = 18
  sns.set_style("darkgrid")

  fig = plt.figure(figsize=(5,4))
  M_rst = []
  n_comp = results["n_comp"]
  M_test_error = np.asarray(results[inputstr]/deno)
  for idx, k in enumerate(n_comp):
    for v in M_test_error[idx]:
      M_rst.append([k, v])

  df = pd.DataFrame(
      data=M_rst,
      index=None,
      columns=["# comp", inputstr])
  avg_test_error = M_test_error.mean(axis=1)
  ax = sns.lineplot(x="# comp", y=inputstr, markers=True, data=df)

  idx_min = np.argmin(avg_test_error)
  
  #print('min:k=%d,mse=%f'%(n_comp[idx_min], avg_test_error[idx_min]))

  if inputstr == 'test_error':
    yl = "Normalized CV MSE"
  else:
    yl = "Normalized train MSE"
  plt.ylabel(yl, fontsize=size_label)
  plt.xlabel("# components (k)", fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  plt.xlim([1, 4])#TODO
  #plt.ylim([0.55,0.95])

  plt.show()