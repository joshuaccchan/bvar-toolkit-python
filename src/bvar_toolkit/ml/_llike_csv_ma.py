import numpy as np
from scipy.linalg import solve_triangular

from .._linalg import ma_solve


def llike_csv_ma(psi, U, Sig, h):
    """bvt.ml.llike_csv_ma - log likelihood of the BVAR-CSV-MA(1) model at MA
    coefficient psi, given untransformed residuals U, error covariance Sig and
    the log-volatility path h (common stochastic volatility exp(h_t) scaling):
    includes the -n/2*sum(h) volatility normalizing term and the first
    observation's (1+psi^2) initialization variance.

    The t-model reduced runs reuse this function with h := log(lam) and with U
    pre-scaled by sqrt(lam).

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    U = np.asarray(U, dtype=float)
    h = np.asarray(h, dtype=float).ravel()
    T, n = U.shape
    Utld = ma_solve(psi, U)
    CSig = np.linalg.cholesky(np.asarray(Sig, dtype=float))
    c = -T * n / 2 * np.log(2 * np.pi) - T * np.sum(np.log(np.diag(CSig))) \
        - n / 2 * np.log(1 + psi**2) - n / 2 * np.sum(h)
    s2 = np.sum(solve_triangular(CSig, Utld.T, lower=True)**2, axis=0)
    return c - .5 * (s2[0] / ((1 + psi**2) * np.exp(h[0])) + np.sum(s2[1:] / np.exp(h[1:])))
