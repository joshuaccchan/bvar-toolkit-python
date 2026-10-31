import numpy as np
from scipy.linalg import solve


def impact_B0(Y0, Y, kappa):
    """bvt.priors.impact_B0 - data-based prior variances for the free elements of
    the impact matrix B0 (AR(4) residual-variance scaling, Minnesota-style).

      beta0, Vbeta = bvt.priors.impact_B0(Y0, Y, kappa)

      Y0    : presample rows; the last 4 are prepended to Y for the univariate
              AR(4) fits that produce the scaling variances sig2
      Y     : T x n estimation sample
      kappa : scalar shrinkage on the impact-matrix block
      beta0 : prior mean of the n(n-1)/2 free elements, zeros
      Vbeta : their prior variances, kappa*sig2_i/sig2_j, stacked equation by
              equation (row 2 of B0 first, then row 3, ...)

    Do not merge with the other impact-matrix priors in the library,
    bvt.priors.vtheta (Valp) and the kappa[2]/sig2 term inside bvt.priors.acp_stru:
    they are numerically different priors on the same object.

    See:
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for
    Large Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    k_beta = n * (n - 1) // 2
    beta0 = np.zeros(k_beta)
    Vbeta = np.zeros(k_beta)
    sig2 = np.zeros(n)
    tmpY = np.vstack([Y0[-4:], Y])
    for i in range(n):
        Z = np.column_stack([np.ones(T), tmpY[3:-1, i], tmpY[2:-2, i], tmpY[1:-3, i],
                             tmpY[:-4, i]])
        tmpb = solve(Z.T @ Z, Z.T @ tmpY[4:, i], assume_a='pos')
        sig2[i] = np.mean((tmpY[4:, i] - Z @ tmpb)**2)
    count_beta = 0
    for i in range(1, n):
        Vbeta[count_beta:count_beta + i] = kappa * np.repeat(sig2[i], i) / sig2[:i]
        count_beta += i
    return beta0, Vbeta
