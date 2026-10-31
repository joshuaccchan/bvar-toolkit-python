import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

from .._rng import as_rng


def var_coef_joint(Y, X, Sig, w, iVb, rng=None):
    """bvt.samplers.var_coef_joint - joint draw of all VAR coefficients under an independent
    normal prior with mean zero, given the error covariance and a volatility weight for
    each period.

      A, Ahat = bvt.samplers.var_coef_joint(Y, X, Sig, w, iVb)
      A, Ahat = bvt.samplers.var_coef_joint(Y, X, Sig, w, iVb, rng)

      Y, X : T x n left-hand side and T x k regressors, as bvt.util.build_lags returns them
      Sig  : n x n error covariance scale, e_t ~ N(0, Sig/w_t)
      w    : T-vector of weights, exp(-h_t) under common stochastic volatility
      iVb  : kn x kn prior precision of vec(A), sparse or dense
      rng  : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      A    : k x n draw, y_t' = x_t' @ A + e_t'
      Ahat : k x n posterior mean

    One call consumes kn standard normals. Written for this toolkit.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    k, n = X.shape[1], Y.shape[1]
    XW = X * np.ravel(w)[:, None]
    iSig = np.linalg.solve(Sig, np.eye(n))
    iSig = (iSig + iSig.T) / 2
    Kb = np.kron(iSig, XW.T @ X)
    Kb = Kb + (iVb.toarray() if sp.issparse(iVb) else np.asarray(iVb, dtype=float))
    Cb = np.linalg.cholesky(Kb)
    cb = (XW.T @ Y @ iSig).ravel(order='F')
    v = solve_triangular(Cb, cb, lower=True)
    u = as_rng(rng).randn(k * n)
    A = solve_triangular(Cb, v + u, lower=True, trans='T').reshape((k, n), order='F')
    Ahat = solve_triangular(Cb, v, lower=True, trans='T').reshape((k, n), order='F')
    return A, Ahat
