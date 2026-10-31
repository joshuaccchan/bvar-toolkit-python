import numpy as np


def vtheta(idx_kappa1, idx_kappa2, kappa, C, sig2):
    """bvt.priors.vtheta - conditional Minnesota-type prior variances of the VAR
    coefficients (Vbeta) and the free elements of the impact matrix (Valp), given
    the shrinkage hyperparameters kappa and the AR(4) residual variances sig2.

      Valp, Vbeta = bvt.priors.vtheta(idx_kappa1, idx_kappa2, kappa, C, sig2)

      idx_kappa1, idx_kappa2, C : the own-lag positions, other-lag positions and
               second moments returned by bvt.priors.minnesota_C
      kappa  : ALWAYS a 4-vector - kappa[0] own lags, kappa[1] other lags,
               kappa[2] impact matrix, kappa[3] intercepts. Callers that fix the
               last two (the hybrid TVP-VAR setting kappa3 = .2, kappa4 = 1) pass
               [kappa1, kappa2, .2, 1].
      sig2   : n-vector of AR(4) residual variances
      Vbeta  : prior variances of the VAR coefficients, stacked equation by
               equation, intercept first in each block
      Valp   : prior variances of the n(n-1)/2 free elements of the impact matrix,
               kappa[2]*sig2_i/sig2_j. A caller wanting only Vbeta may pass
               kappa[2] = NaN and take Vbeta; Valp is then NaN throughout.

    See:
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for
    Large Bayesian VARs, International Journal of Forecasting, 37(3): 1212-1226
    """
    kappa = np.asarray(kappa, dtype=float).ravel()
    C = np.asarray(C, dtype=float).ravel()
    sig2 = np.asarray(sig2, dtype=float).ravel()
    np_ = len(idx_kappa1)
    n = len(idx_kappa2) // np_ + 1
    Vbeta = np.zeros(C.size)
    Valp = np.zeros(n * (n - 1) // 2)

    Vbeta[::np_ + 1] = kappa[3] * sig2                    # intercepts
    Vbeta[idx_kappa1] = kappa[0] * C[idx_kappa1]          # own lags
    Vbeta[idx_kappa2] = kappa[1] * C[idx_kappa2]          # other lags

    count_alp = 0
    for ii in range(n):
        Valp[count_alp:count_alp + ii] = kappa[2] * np.repeat(sig2[ii], ii) / sig2[:ii]
        count_alp += ii
    return Valp, Vbeta
