import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve

VARIANTS = ('largebvar_nc', 'mlvarsv_ncp', 'opthyper_ncp', 'kron_script')


def niw(p, kappa, Y0, Y, variant):
    """bvt.priors.niw - natural-conjugate (normal-inverse-Wishart) prior for a
    VAR(p) with intercept, in four legacy parameterizations.

      A0, VA0, nu0, S0, U_hat = bvt.priors.niw(p, kappa, Y0, Y, variant)

      p       : lag length
      kappa   : shrinkage hyperparameters; length and meaning set by variant
      Y0, Y   : presample rows and the T x n estimation sample. The univariate
                AR(4) fits prepend Y0[-p:], whose design matrix is
                conformable only for p = 4; 'mlvarsv_ncp' prepends the last 4
                rows whatever p is
      variant : one of the four below. They are numerically different; never
                merge them
        'largebvar_nc'  kappa = [c1, c2], the coefficient and the intercept scale;
                        VA0 a dense k-vector; nu0 = n+3; S0 = diag(sig2)
        'mlvarsv_ncp'   as 'largebvar_nc', except for the presample block above
        'opthyper_ncp'  kappa = [kappa1, ..., kappa5]: coefficient scale, lag decay
                        exponent, intercept scale, then nu0 = kappa[3]+n+1 and
                        S0 = kappa[4]*diag(sig2). VA0 is a SPARSE k x k diagonal
                        matrix here and a k-vector in the other variants
        'kron_script'   as 'largebvar_nc', except S0 = eye(n)
      A0      : k x n prior mean of the coefficient matrix, zeros, k = 1+n*p
      VA0     : prior variances of the coefficients; vector or sparse matrix,
                see variant
      nu0, S0 : inverse-Wishart degrees of freedom and scale matrix
      U_hat   : T x n univariate AR(4) residuals, sig2 their mean squares

    See:
    Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds),
    Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for
    Large Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
    Chan, J. C. C., L. Jacobi, and D. Zhu (2020). Efficient Selection of
    Hyperparameters in Large Bayesian VARs Using Automatic Differentiation,
    Journal of Forecasting, 39(6): 934-943.
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics,
    38(1), 68-79.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}'; use largebvar_nc, mlvarsv_ncp, "
                         'opthyper_ncp or kron_script')
    kappa = np.asarray(kappa, dtype=float).ravel()
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    p = int(p)
    Tt, n = Y.shape
    k = 1 + n * p
    A0 = np.zeros((k, n))
    VA0 = np.zeros(k)
    sig2 = np.zeros(n)
    U_hat = np.zeros((Tt, n))
        # construct VA0
    tmpY = np.vstack([Y0[-4:], Y]) if variant == 'mlvarsv_ncp' else np.vstack([Y0[-p:], Y])
    for i in range(n):
        Z = np.column_stack([np.ones(Tt), tmpY[3:-1, i], tmpY[2:-2, i], tmpY[1:-3, i],
                             tmpY[:-4, i]])
        tmpb = solve(Z.T @ Z, Z.T @ tmpY[4:, i], assume_a='pos')
        U_hat[:, i] = tmpY[4:, i] - Z @ tmpb
        sig2[i] = np.mean(U_hat[:, i]**2)
    for i in range(k):
        lag = -(-i // n)                          # ceil(i/n)
        idx = (i - 1) % n                         # variable index
        if i == 0:                                # intercept
            VA0[0] = kappa[2] if variant == 'opthyper_ncp' else kappa[1]
        elif variant == 'opthyper_ncp':
            VA0[i] = kappa[0] / (lag**kappa[1] * sig2[idx])
        else:
            VA0[i] = kappa[0] / (lag**2 * sig2[idx])
    if variant == 'opthyper_ncp':
        S0 = kappa[4] * np.diag(sig2)
        nu0 = kappa[3] + n + 1
        VA0 = sp.diags_array(VA0, format='csc')
    else:
        nu0 = n + 3
        S0 = np.eye(n) if variant == 'kron_script' else np.diag(sig2)
    return A0, VA0, nu0, S0, U_hat
