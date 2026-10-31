import numpy as np


def resid_var_allvars_ridge(Y0, Y):
    """bvt.priors.resid_var_allvars_ridge - residual variances from regressing each
    variable on 4 lags of ALL variables (VAR(4)-style regressor set) with a 1e-4
    ridge on the normal equations; used to set the Minnesota-prior scalings in the
    hybrid TVP-VAR.

      sig2 = bvt.priors.resid_var_allvars_ridge(Y0, Y)

      Y0   : presample rows; the last 4 are prepended to Y
      Y    : T x n estimation sample
      sig2 : n-vector of residual variances

    NEVER merge with bvt.priors.resid_var_ar4: that one runs univariate AR(4)
    regressions with no ridge - numerically different sig2, hence different
    Minnesota scalings.

    See:
    Chan, J.C.C. (2023). Large Hybrid Time-Varying Parameter VARs, Journal of
    Business and Economic Statistics, 41(3): 890-905
    """
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    tmpY = np.vstack([Y0[-4:], Y])
    Z = np.column_stack([np.ones(T), tmpY[3:-1], tmpY[2:-2], tmpY[1:-3], tmpY[:-4]])
    ZZ = Z.T @ Z + 1e-4 * np.eye(Z.shape[1])
    sig2 = np.zeros(n)
    for i in range(n):
        tmpb = np.linalg.solve(ZZ, Z.T @ tmpY[4:, i])
        sig2[i] = np.mean((tmpY[4:, i] - Z @ tmpb)**2)
    return sig2
