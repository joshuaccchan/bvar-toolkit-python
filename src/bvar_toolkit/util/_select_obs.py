import numpy as np
import scipy.sparse as sp


def select_obs(Y):
    """bvt.util.select_obs - the selection matrices of a missing data pattern,
    y = So @ yo + Sm @ ym, splitting the stacked data into its observed and missing values.

      So, Sm, yo = bvt.util.select_obs(Y)

      Y  : T x n data matrix, NaN wherever the value is missing
      So : Tn x No sparse selection matrix (csc_array) for the observed values
      Sm : Tn x Nm sparse selection matrix (csc_array) for the missing values
      yo : No-vector of observed values, ordered as So takes them

    y = (y_1', ..., y_T')' = Y.ravel() stacks the n variables within each period, and yo and ym
    follow that order, so the precision of ym given yo is banded whenever the precision of y is.
    [So, Sm] is a permutation matrix, so Sm.T @ y recovers the missing values of a completed y.
    An all-observed or all-missing pattern gives the corresponding matrix Tn x 0.

    See:
    Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian State
    Space Models with Missing Data, Journal of Econometrics, 236(1): 105468, Section 2.1.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.size == 0:
        raise ValueError('Y must be a nonempty T x n numeric matrix')
    T, n = Y.shape
    y = Y.reshape(T * n)                            # y = (y_1', ..., y_T')'
    io = np.flatnonzero(~np.isnan(y))
    im = np.flatnonzero(np.isnan(y))
    So = sp.csc_array((np.ones(io.size), (io, np.arange(io.size))), shape=(T * n, io.size))
    Sm = sp.csc_array((np.ones(im.size), (im, np.arange(im.size))), shape=(T * n, im.size))
    return So, Sm, y[io]
