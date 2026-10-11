import numpy as np
import scipy.sparse as sp


def surform2(X, n):
    """bvt.util.surform2 - sparse SUR expansion with rows kron(I_n, X[t]) for stacked-vector VAR
    sampling.

      Xout = bvt.util.surform2(X, n)

      X    : T x k matrix of regressors
      n    : number of equations
      Xout : T*n x n*k sparse matrix (csc_array) whose tth block of n rows is kron(I_n, X[t]),
             so that Xout @ vec(A) returns the fitted values stacked as Y.ravel(), where A is
             the k x n coefficient matrix

    NOT the same operator as bvt.util.surform, which is the T x Tk block-diagonal expansion
    used for TVP state stacking.

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large Bayesian
    VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    X = np.asarray(X, dtype=float)
    r, c = X.shape
    n = int(n)
    vals = np.repeat(X, n, axis=0).reshape(-1)
    rows = np.repeat(np.arange(r * n), c)
    cols = np.tile(np.arange(n * c), r)
    Xout = sp.csc_array((vals, (rows, cols)), shape=(r * n, n * c))
    Xout.eliminate_zeros()
    return Xout
