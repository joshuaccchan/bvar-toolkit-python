import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def eq_gauss(Y, Z, h, Valp, Vbeta, rng=None):
    """bvt.samplers.eq_gauss - one full equation-by-equation sweep of the Gaussian
    coefficient draw in the structural-form BVAR of Chan (2021, IJF): for each
    equation ii = 0..n-1, draw thetai = (beta_ii; alp_ii) jointly from its Gaussian
    conditional through the precision
      Kthetai = iVthetai + Xi.T @ diag(exp(-h[:, ii])) @ Xi,   Xi = [Z, -Y[:, :ii]],
    via its lower Cholesky factor, and accumulate the structural residuals U.

      beta, alp, U = bvt.samplers.eq_gauss(Y, Z, h, Valp, Vbeta)
      beta, alp, U = bvt.samplers.eq_gauss(Y, Z, h, Valp, Vbeta, rng)

    Inputs:  Y     - T x n observations (equation ii regresses Y[:, ii] on
                     [Z, -Y[:, :ii]], the structural triangular form)
             Z     - T x (n*p+1) design [1, y_{t-1}, ..., y_{t-p}]
             h     - T x n log-volatilities
             Valp  - n*(n-1)/2 prior variances of the impact-matrix elements
             Vbeta - n*(n*p+1) prior variances of the VAR coefficients
             rng   - None, an integer seed, a numpy.random.Generator or a bvt.Stream
    Outputs: beta  - n*(n*p+1) stacked coefficient draw
             alp   - n*(n-1)/2 impact-matrix draw
             U     - T x n structural residuals yi - Xi @ thetai

    Valp and Vbeta are used exactly as passed. Any scaling of the prior variances
    - some callers double both before this step - stays with the caller.

    See:
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for
    Large Bayesian VARs, International Journal of Forecasting, 37(3), 1212-1226.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    h = np.asarray(h, dtype=float)
    Valp = np.asarray(Valp, dtype=float).ravel()
    Vbeta = np.asarray(Vbeta, dtype=float).ravel()
    T, n = Y.shape
    np_ = Z.shape[1] - 1                                 # = n*p
    beta = np.zeros(n * (np_ + 1))
    alp = np.zeros(n * (n - 1) // 2)
    count_alp = 0
    U = np.zeros((T, n))
    for ii in range(n):
        yi = Y[:, ii]
        ki = np_ + 1 + ii
        Xi = np.column_stack([Z, -Y[:, :ii]])
        iVthetai = 1 / np.r_[Vbeta[ii * (np_ + 1):(ii + 1) * (np_ + 1)],
                             Valp[count_alp:count_alp + ii]]
        XiiSighi = Xi.T * np.exp(-h[:, ii])
        Kthetai = XiiSighi @ Xi
        Kthetai[np.diag_indices(ki)] += iVthetai
        CKthetai = np.linalg.cholesky(Kthetai)
        thetai_hat = solve_triangular(CKthetai, solve_triangular(CKthetai, XiiSighi @ yi,
                                                                 lower=True),
                                      lower=True, trans='T')
        thetai = thetai_hat + solve_triangular(CKthetai, r.randn(ki), lower=True, trans='T')
        U[:, ii] = yi - Xi @ thetai
        beta[ii * (np_ + 1):(ii + 1) * (np_ + 1)] = thetai[:np_ + 1]
        alp[count_alp:count_alp + ii] = thetai[np_ + 1:]
        count_alp += ii
    return beta, alp, U
