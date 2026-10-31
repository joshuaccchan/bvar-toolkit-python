import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def alp_tri_cs(E, h, Valp, o=None, rng=None):
    """bvt.samplers.alp_tri_cs - Gaussian draw of the free elements of the
    unit-lower-triangular impact matrix A in the Cholesky / triangularized (CS)
    SVAR-SV model, row by row: for equation ii = 2:n the ii-1 free elements
    alp_ii regress the residual E[:, ii] on -E[:, :ii] weighted by
    exp(-h[:, ii]), under independent N(0, Valp) priors. The prior mean is zero;
    there is no alp0 term.

      alp = bvt.samplers.alp_tri_cs(E, h, Valp)
      alp = bvt.samplers.alp_tri_cs(E, h, Valp, o)   # outlier-scaled (VAR-SVO)

      E    : T x n residual matrix Y - XB; the CALLER computes it and the caller
             writes the draw back into A with A[np.tril_indices(n, -1)] = alp
      h    : T x n log-volatilities
      Valp : n(n-1)/2 stacked prior variances, rows ordered (2,1),
             (3,1),(3,2), (4,1),... - row-major lower triangle
      o    : T-vector of outlier scales, optional; default ones(T)
      rng  : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      alp  : n(n-1)/2-vector of draws

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    r = as_rng(rng)
    E = np.asarray(E, dtype=float)
    h = np.asarray(h, dtype=float)
    Valp = np.asarray(Valp, dtype=float).ravel()
    T, n = E.shape
    o = np.ones(T) if o is None or np.size(o) == 0 else np.asarray(o, dtype=float).ravel()
    alp = np.zeros(n * (n - 1) // 2)
    count_alp = 0
    for ii in range(1, n):
        X_alpi = -E[:, :ii]
        iD = np.exp(-h[:, ii]) / o**2
        Kalpi = X_alpi.T @ (iD[:, None] * X_alpi)
        Kalpi[np.diag_indices(ii)] += 1 / Valp[count_alp:count_alp + ii]
        CKalpi = np.linalg.cholesky(Kalpi)
        alpi_hat = solve_triangular(CKalpi, solve_triangular(CKalpi, X_alpi.T @ (iD * E[:, ii]),
                                                             lower=True), lower=True, trans='T')
        alp[count_alp:count_alp + ii] = alpi_hat + solve_triangular(CKalpi, r.randn(ii),
                                                                    lower=True, trans='T')
        count_alp += ii
    return alp
