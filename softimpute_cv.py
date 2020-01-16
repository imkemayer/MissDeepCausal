import numpy as np
from sklearn.utils.extmath import randomized_svd

F32PREC = np.finfo(np.float32).eps


def converged(x_old, x, mask, thresh):
  x_old_na = x_old[mask]
  x_na = x[mask]
  difference = x_old_na - x_na
  mse = np.sum(difference ** 2)
  denom = np.sqrt((x_old_na ** 2).sum())
  
  if denom == 0 or (denom < F32PREC and np.sqrt(mse) > F32PREC):
      return False
  else:
      return (np.sqrt(mse) / denom) < thresh

def softimpute(x, lamb, maxit = 1000, thresh = 1e-5):
  mask = ~np.isnan(x)
  imp = x.copy()
  imp[~mask] = 0
  for i in range(maxit):
    U, d, V = np.linalg.svd(imp, compute_uv = True)
    d_thresh = np.maximum(d - lamb, 0)
    rank = (d_thresh > 0).sum()
    d_thresh = d_thresh[:rank]
    U_thresh = U[:, :rank]
    V_thresh = V[:rank, :]
    D_thresh = np.diag(d_thresh)
    res = np.dot(U_thresh, np.dot(D_thresh, V_thresh))
    if converged(imp, res, mask, thresh):
      break
    imp[~mask] = res[~mask]
    
  return U_thresh, res



def cv_softimpute(x, grid_len = 15, maxit = 1000, thresh = 1e-5):
  # impute with constant
  mask = ~np.isnan(x)
  x0 = x.copy()
  x0[~mask] = 0
  # svd on x0
  d = np.linalg.svd(x0, compute_uv = False)
  # generate grid for lambda values
  lambda_max = np.max(d)
  lambda_min = 0.001*lambda_max
  grid_lambda = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), grid_len).tolist())

  def test_x(x, mask):
    # generate additional missing values
    mmask = np.array(np.random.binomial(np.ones_like(mask), mask * .05), dtype=bool)
    xx = x.copy()
    xx[mmask] = np.nan
    return xx, mmask

  cv_error = []
  for lamb in grid_lambda:
    xx, mmask = test_x(x, mask)
    mmask = ~np.isnan(xx)
    _, res = softimpute(xx, lamb, maxit, thresh)
    cv_error.append(np.sqrt(np.nanmean((res.flatten() - x.flatten())**2)))

  return cv_error, grid_lambda
  