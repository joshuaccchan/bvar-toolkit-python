import numpy as np
from scipy.linalg import solve_triangular

from .._linalg import ma_solve


def llike_ma(psi, U, Sig):
    """bvt.ml.llike_ma - log likelihood of the homoskedastic BVAR-MA(1) model at
    MA coefficient psi, given the untransformed residuals U = Y - X @ A and error
    covariance Sig: u_t = eps_t + psi*eps_{t-1}, eps_t ~ N(0,Sig), with the
    first transformed observation carrying the initialization variance factor
    (1+psi^2) and the matching -n/2*log(1+psi^2) constant.

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    U = np.asarray(U, dtype=float)
    T, n = U.shape
    Utld = ma_solve(psi, U)
    CSig = np.linalg.cholesky(np.asarray(Sig, dtype=float))
    c = -T * n / 2 * np.log(2 * np.pi) - T * np.sum(np.log(np.diag(CSig))) \
        - n / 2 * np.log(1 + psi**2)
    s2 = np.sum(solve_triangular(CSig, Utld.T, lower=True)**2, axis=0)
    return c - .5 * (s2[0] / (1 + psi**2) + np.sum(s2[1:]))
