import numpy as np


def minnesota_C(n, p, sig2):
    """bvt.priors.minnesota_C - Minnesota-prior second-moment building blocks C and
    the index sets of the own-lag (kappa1) and other-lag (kappa2) coefficients,
    equation by equation (intercept first, then lag-1 block, lag-2 block, ...).

      C, idx_kappa1, idx_kappa2 = bvt.priors.minnesota_C(n, p, sig2)

      n, p       : number of variables and lag length
      sig2       : n-vector of AR(4) residual variances
      C          : (n^2 p + n)-vector of second moments - sig2_i on the
                   intercepts, 1/l^2 on own lags, sig2_i/(l^2*sig2_j) on other
                   lags. The shrinkage hyperparameters scale it in
                   bvt.priors.vtheta.
      idx_kappa1 : positions in C of the own-lag coefficients, counted from 0
      idx_kappa2 : positions in C of the other-lag coefficients, counted from 0

    See:
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for
    Large Bayesian VARs, International Journal of Forecasting, 37(3): 1212-1226
    """
    sig2 = np.asarray(sig2, dtype=float).ravel()
    n, p = int(n), int(p)
    C = np.zeros(n * n * p + n)
    idx_kappa1 = []
    idx_kappa2 = []
    count = 0
    for ii in range(n):
        Ci = np.zeros(n * p + 1)
            # construct Ci
        for j in range(n * p + 1):
            lag = -(-j // n)                      # lag length, ceil(j/n)
            idx = (j - 1) % n                     # variable index
            if j == 0:                            # intercept
                Ci[j] = sig2[ii]
            elif idx == ii:                       # own lag
                Ci[j] = 1 / lag**2
                idx_kappa1.append(count)
            else:                                 # lag of other variables
                Ci[j] = sig2[ii] / (lag**2 * sig2[idx])
                idx_kappa2.append(count)
            count += 1
        C[ii * (n * p + 1):(ii + 1) * (n * p + 1)] = Ci
    return C, np.array(idx_kappa1, dtype=int), np.array(idx_kappa2, dtype=int)
