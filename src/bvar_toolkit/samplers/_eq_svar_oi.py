import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def eq_svar_oi(Y, X, B0, h, A, tmpdV, rng=None):
    """bvt.samplers.eq_svar_oi - equation-by-equation Gaussian draw of the VAR
    coefficient matrix A (k x n, intercept first) in the ORDER-INVARIANT
    structural SVAR-SV model, one column A[:, ii] at a time from its Gaussian
    conditional. Prior variances enter as the stacked k*n vector tmpdV, which the
    Vbeta output of bvt.priors.vtheta provides.

      A = bvt.samplers.eq_svar_oi(Y, X, B0, h, A, tmpdV)
      A = bvt.samplers.eq_svar_oi(Y, X, B0, h, A, tmpdV, rng)

    rng is None, an integer seed, a numpy.random.Generator or a bvt.Stream. The input A
    is not modified.

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    A = np.array(A, dtype=float)
    tmpdV = np.asarray(tmpdV, dtype=float).ravel()
    k = X.shape[1]
    n = Y.shape[1]
    Lambda = np.exp(np.asarray(h, dtype=float) / 2).ravel(order='F')
    for ii in range(n):
        A[:, ii] = 0
        yi = ((Y - X @ A) @ B0.T).ravel(order='F') / Lambda
        Wi = np.kron(B0[:, ii][:, None], X) / Lambda[:, None]
        Kalphai = Wi.T @ Wi
        Kalphai[np.diag_indices(k)] += 1 / tmpdV[ii * k:(ii + 1) * k]
        CKalphai = np.linalg.cholesky(Kalphai)
        alphai_hat = solve_triangular(CKalphai, solve_triangular(CKalphai, Wi.T @ yi, lower=True),
                                      lower=True, trans='T')
        A[:, ii] = alphai_hat + solve_triangular(CKalphai, r.randn(k), lower=True, trans='T')
    return A
