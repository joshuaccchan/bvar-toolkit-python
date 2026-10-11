import numpy as np


def build_lags(Yfull, p):
    """bvt.util.build_lags - lagged design matrix with the toolkit-wide convention: intercept
    first, then the lag-1 block, ..., then the lag-p block.

      Y, Z = bvt.util.build_lags(Yfull, p)

      Yfull : T0 x n data matrix (the first p rows serve as initial conditions); a 1-D array is
              one series
      p     : lag length, a positive integer strictly less than T0
      Y     : (T0-p) x n left-hand-side observations, Yfull[p:]
      Z     : (T0-p) x (1+n*p) regressor matrix [1, y_{t-1}, ..., y_{t-p}]

    If a driver holds its initial conditions separately as Y0, pass them in the same matrix:
    build_lags(np.vstack([Y0[-p:], Y]), p).
    """
    Yfull = np.asarray(Yfull, dtype=float)
    if Yfull.ndim == 1:
        Yfull = Yfull[:, None]
    T0, n = Yfull.shape
    if p < 1 or p != int(p):
        raise ValueError('p must be a positive integer')
    p = int(p)
    if T0 <= p:
        raise ValueError(f'need more than p = {p} rows of data')
    T = T0 - p
    Y = Yfull[p:].copy()
    Z = np.ones((T, 1 + n * p))
    for ii in range(1, p + 1):
        Z[:, 1 + (ii - 1) * n:1 + ii * n] = Yfull[p - ii:T0 - ii]
    return Y, Z
