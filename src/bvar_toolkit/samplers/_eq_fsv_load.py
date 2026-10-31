import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def eq_fsv_load(Y, X, F, h, A, L, Valp, alp0, Vl, l0, rng=None):
    """bvt.samplers.eq_fsv_load - equation-by-equation Gaussian draw of the VAR
    coefficients A (k x n) jointly with the free factor loadings L (n x r) in the
    factor-SV VAR, conditional on the factor path F and the log-volatilities h.
    Equation ii regresses Y[:, ii] on [X F[:, :min(ii,r)]] weighted by
    exp(-h[:, ii]); the first r equations are normalized (unit diagonal, so
    F[:, ii] is subtracted from the left-hand side instead of loaded).

      A, L = bvt.samplers.eq_fsv_load(Y, X, F, h, A, L, Valp, alp0, Vl, l0)
      A, L = bvt.samplers.eq_fsv_load(Y, X, F, h, A, L, Valp, alp0, Vl, l0, rng)

      Valp, alp0 : k*n stacked prior variances / means of A
      Vl, l0     : scalar prior variance / mean of every free loading
      rng        : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    The inputs A and L are not modified.

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    rg = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float).reshape(Y.shape[0], -1)
    h = np.asarray(h, dtype=float)
    A = np.array(A, dtype=float)
    L = np.array(L, dtype=float).reshape(Y.shape[1], -1)
    Valp = np.asarray(Valp, dtype=float).ravel()
    alp0 = np.asarray(alp0, dtype=float).ravel()
    n = Y.shape[1]
    k = X.shape[1]
    r = F.shape[1]
    for ii in range(n):
        m = min(ii, r)                                 # free loadings of this equation
        Zi = np.hstack([X, F[:, :m]])
        iVthetai = 1 / np.r_[Valp[ii * k:(ii + 1) * k], Vl * np.ones(m)]
        thetai0 = np.r_[alp0[ii * k:(ii + 1) * k], l0 * np.ones(m)]
        ZiSigi = Zi.T * np.exp(-h[:, ii])
        lhs = Y[:, ii] - F[:, ii] if ii < r else Y[:, ii]
        dthetai = iVthetai * thetai0 + ZiSigi @ lhs
        Kthetai = ZiSigi @ Zi
        Kthetai[np.diag_indices(k + m)] += iVthetai
        CK = np.linalg.cholesky(Kthetai)
        thetai_hat = solve_triangular(CK, solve_triangular(CK, dthetai, lower=True), lower=True,
                                      trans='T')
        thetai = thetai_hat + solve_triangular(CK, rg.randn(k + m), lower=True, trans='T')
        A[:, ii] = thetai[:k]
        L[ii, :m] = thetai[k:]
    return A, L
