import numpy as np

from .._struct import Struct


def acp_stru(n, p, kappa, sig2, idx_ns=None):
    """bvt.priors.acp_stru - asymmetric conjugate prior elicited directly on the structural
    parameterization of the VAR.

      prior = bvt.priors.acp_stru(n, p, kappa, sig2)
      prior = bvt.priors.acp_stru(n, p, kappa, sig2, idx_ns)

      n, p   : number of variables and lag length
      kappa  : [kappa1 kappa2 kappa3 kappa4] - own lags, other lags, impact matrix, intercepts
      sig2   : n-vector of AR(4) residual variances (bvt.priors.resid_var_ar4)
      idx_ns : indices of nonstationary variables, counted from 0, whose first own lag gets
               prior mean one (default none)
      prior  : Struct with fields beta0, Vbeta (VAR coefficients, (n p + 1) x n, equation by
               equation), alp0, Valp (free elements of the impact matrix, stacked equation by
               equation), nu, S (the n inverse-gamma error-variance parameters)

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169
    """
    idx_ns = [] if idx_ns is None else [int(i) for i in np.ravel(idx_ns)]
    sig2 = np.asarray(sig2, dtype=float).ravel()
    kb = n * p + 1                                  # coefficients per equation
    k_alp = n * (n - 1) // 2
    beta0 = np.zeros((kb, n))
    alp0 = np.zeros(k_alp)
    Vbeta = np.zeros((kb, n))
    Valp = np.zeros(k_alp)
    nu = np.zeros(n)
    S = np.zeros(n)
    count_alp = 0
    for ii in range(n):
        mi, Vi, nui, Si = _prior_acpi(n, p, ii + 1, kappa, sig2, ii in idx_ns)
        beta0[:, ii] = mi[:kb]
        alp0[count_alp:count_alp + ii] = mi[kb:]
        Vbeta[:, ii] = Vi[:kb]
        Valp[count_alp:count_alp + ii] = Vi[kb:]
        nu[ii] = nui
        S[ii] = Si
        count_alp += ii
    return Struct(beta0=beta0, alp0=alp0, Vbeta=Vbeta, Valp=Valp, nu=nu, S=S)


def _prior_acpi(n, p, var_i, kappa, sig2, is_ns):
    """Prior of equation var_i (counted from 1, as in the paper)."""
    ki = var_i + n * p
    mi = np.zeros(ki)
    Vi = np.zeros(ki)
    for j in range(1, ki + 1):
        if j <= n * p + 1:
            lag = int(np.ceil((j - 1) / n))         # lag length
            idx = (j - 1) % n                       # variable, counted from 1 below
            if idx == 0:
                idx = n
        else:
            lag = 0
            idx = j - (n * p + 1)
        if j == 1:                                  # intercept
            Vi[j - 1] = kappa[3]
        elif j > n * p + 1:                         # alpha_i
            Vi[j - 1] = kappa[2] / sig2[idx - 1]
        elif idx == var_i:                          # own lag
            Vi[j - 1] = kappa[0] / (lag ** 2 * sig2[idx - 1])
            if lag == 1 and is_ns:                  # first own lag of a nonstationary variable
                mi[j - 1] = 1
        else:                                       # lag of another variable
            Vi[j - 1] = kappa[1] / (lag ** 2 * sig2[idx - 1])
    return mi, Vi, 1 + var_i / 2, sig2[var_i - 1] / 2
