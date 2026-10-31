import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def eq_var_redu_tri(Y, X, B0, h, A, Valp, alp0, o=None, rng=None):
    """bvt.samplers.eq_var_redu_tri - equation-by-equation Gaussian draw of the VAR
    coefficient matrix A (k x n, intercept first) in the reduced-form VAR-SV with
    a unit-lower-triangular impact matrix B0. Equation ii stacks only rows ii:n of
    the rotated system and draws A[:, ii] from its Gaussian conditional.

      A = bvt.samplers.eq_var_redu_tri(Y, X, B0, h, A, Valp, alp0)      # VAR-SV
      A = bvt.samplers.eq_var_redu_tri(Y, X, B0, h, A, Valp, alp0, o)   # VAR-SVO

      Valp : k*n stacked prior variances
      alp0 : k*n stacked prior means
      o    : T-vector of outlier scales, optional; default ones(T)
      rng  : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    The input A is not modified.

    Sibling of bvt.samplers.eq_svar_oi (order-invariant SVAR-SV), which stacks all
    n rows, has no prior-mean term and no o. Do not merge the two.

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    h = np.asarray(h, dtype=float)
    A = np.array(A, dtype=float)
    Valp = np.asarray(Valp, dtype=float).ravel()
    alp0 = np.asarray(alp0, dtype=float).ravel()
    T, n = Y.shape
    k = X.shape[1]
    o = np.ones(T) if o is None or np.size(o) == 0 else np.asarray(o, dtype=float).ravel()
    sqrt_exph = np.exp(h / 2)
    for ii in range(n):
        A[:, ii] = 0
        Lambda = (sqrt_exph[:, ii:] * o[:, None]).ravel(order='F')
        yi = ((Y - X @ A) @ B0[ii:].T).ravel(order='F') / Lambda
        Wi = np.kron(B0[ii:, ii][:, None], X) / Lambda[:, None]
        iValpi = 1 / Valp[ii * k:(ii + 1) * k]
        Kalpi = Wi.T @ Wi
        Kalpi[np.diag_indices(k)] += iValpi
        CKalpi = np.linalg.cholesky(Kalpi)
        rhs = iValpi * alp0[ii * k:(ii + 1) * k] + Wi.T @ yi
        alpi_hat = solve_triangular(CKalpi, solve_triangular(CKalpi, rhs, lower=True),
                                    lower=True, trans='T')
        A[:, ii] = alpi_hat + solve_triangular(CKalpi, r.randn(k), lower=True, trans='T')
    return A
