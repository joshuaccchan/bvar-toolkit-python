import numpy as np
from scipy.linalg import solve


def resid_var_ar4(Y0, Y):
    """bvt.priors.resid_var_ar4 - residual variances of univariate AR(4) models, used to set
    the Minnesota-prior scalings.

      sig2 = bvt.priors.resid_var_ar4(Y0, Y)

      Y0   : presample rows; the last 4 are prepended to Y
      Y    : T x n estimation sample
      sig2 : n-vector of AR(4) residual variances

    NEVER merge with bvt.priors.resid_var_allvars_ridge: that one regresses each variable on 4
    lags of ALL variables with a 1e-4 ridge - numerically different sig2.

    See:
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for Large Bayesian VARs,
    International Journal of Forecasting, 37(3): 1212-1226
    """
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    sig2 = np.zeros(n)
    tmpY = np.vstack([Y0[-4:], Y])
    for i in range(n):
        Z = np.column_stack([np.ones(T), tmpY[3:-1, i], tmpY[2:-2, i], tmpY[1:-3, i], tmpY[:-4, i]])
        tmpb = solve(Z.T @ Z, Z.T @ tmpY[4:, i], assume_a='pos')
        sig2[i] = np.mean((tmpY[4:, i] - Z @ tmpb) ** 2)
    return sig2
