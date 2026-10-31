import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def eq_var_oi(Y, X, B0, h, A, tmpdV, rng=None):
    """bvt.samplers.eq_var_oi - equation-by-equation Gaussian draw of the VAR
    coefficient matrix A (k x n, intercept first) in the order-invariant VAR-SV
    model
        y_t = A' x_t + u_t,   B0 u_t = eps_t,   eps_t ~ N(0, diag(exp(h_t))),
    the same conditional as bvt.samplers.eq_svar_oi, computed in O(T k^2 + k^3)
    per equation in place of O(T n k^2 + k^3).

      A = bvt.samplers.eq_var_oi(Y, X, B0, h, A, tmpdV)
      A = bvt.samplers.eq_var_oi(Y, X, B0, h, A, tmpdV, rng)

      Y     : T x n observations
      X     : T x k regressors [1, y_{t-1}, ..., y_{t-p}] (bvt.util.build_lags)
      B0    : n x n INVERSE impact matrix, eps_t = B0 u_t, so the impact matrix
              itself is B0^{-1} and the reduced-form covariance is
              B0^{-1} diag(exp(h_t)) B0^{-T}. Pass B0, not B0^{-1}.
      h     : T x n log-variances of the structural innovations, column j = eps_jt
      A     : current k x n coefficients (columns updated in order; the input is not
              modified)
      tmpdV : k*n stacked prior variances, column-major (zero prior mean; to use
              a prior mean A0, call on Y - X @ A0 with A - A0 and add A0 back -
              an exact reparameterization)
      rng   : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    bvt.samplers.eq_svar_oi draws from the same conditional with the same rng
    consumption, rng.randn(k) per equation in order. It follows the code of Chan,
    Koop and Yu (2024) and must never be edited for speed. Use this function in new
    code.

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    h = np.asarray(h, dtype=float)
    A = np.array(A, dtype=float)
    tmpdV = np.asarray(tmpdV, dtype=float).ravel()
    T, n = Y.shape
    k = X.shape[1]
    if h.shape != (T, n):
        raise ValueError(f'h must be {T} x {n} (T x n), not {h.shape[0]} x {h.shape[1]}')
    if B0.shape != (n, n):
        raise ValueError(f'B0 must be {n} x {n}, not {B0.shape[0]} x {B0.shape[1]}')
    if tmpdV.size != k * n:
        raise ValueError(f'tmpdV must have k*n = {k * n} elements, not {tmpdV.size}')
    eh_inv = np.exp(-h)                        # T x n, column j = exp(-h_jt)
    for ii in range(n):
        A[:, ii] = 0
        Etil = (Y - X @ A) @ B0.T              # row t = (B0 (y_t - A_{-ii}' x_t))'
        b = B0[:, ii]                          # loadings of A[:, ii] in the n structural equations
        w = eh_inv @ (b**2)                    # T-vector of precision weights
        c = (eh_inv * Etil) @ b
        iVi = 1 / tmpdV[ii * k:(ii + 1) * k]
        Kai = X.T @ (w[:, None] * X)
        Kai[np.diag_indices(k)] += iVi
        CKai = np.linalg.cholesky(Kai)
        ai_hat = solve_triangular(CKai, solve_triangular(CKai, X.T @ c, lower=True), lower=True,
                                  trans='T')
        A[:, ii] = ai_hat + solve_triangular(CKai, r.randn(k), lower=True, trans='T')
    return A
