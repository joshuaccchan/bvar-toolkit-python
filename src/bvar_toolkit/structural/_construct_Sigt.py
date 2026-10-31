import numpy as np


def construct_Sigt(h, B0):
    """bvt.structural.construct_Sigt - time-varying reduced-form covariance matrices
    Sigt[t] = B0^{-1} diag(exp(h_t)) B0^{-T} implied by an impact matrix B0
    (full n x n in the OI model; unit-lower-triangular A in the CS model) and the
    T x n log-volatility paths h. Pure transform - consumes no rng.

      Sigt = bvt.structural.construct_Sigt(h, B0)

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    h = np.asarray(h, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    Bi = np.linalg.solve(B0, np.eye(B0.shape[0]))
    return np.einsum('ij,tj,kj->tik', Bi, np.exp(h), Bi)
