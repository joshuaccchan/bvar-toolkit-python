import numpy as np

from .._struct import Struct
from ._acp_stru import acp_stru


def acp_redu(n, p, kappa, sig2, idx_ns=None):
    """bvt.priors.acp_redu - asymmetric conjugate prior elicited on the reduced-form
    parameterization, mapped to the implied prior on the structural parameterization.

      prior = bvt.priors.acp_redu(n, p, kappa, sig2)
      prior = bvt.priors.acp_redu(n, p, kappa, sig2, idx_ns)

      n, p   : number of variables and lag length
      kappa  : [kappa1 kappa2 kappa3 kappa4] - own lags, other lags, impact matrix, intercepts
      sig2   : n-vector of AR(4) residual variances (bvt.priors.resid_var_ar4)
      idx_ns : indices of nonstationary variables, counted from 0, whose first own lag gets
               prior mean one (default none)
      prior  : Struct with fields beta0, Vbeta (VAR coefficients, (n p + 1) x n, equation by
               equation), alp0, Valp (free elements of the impact matrix), nu, S (the n
               inverse-gamma error-variance parameters). Only Vbeta differs from
               bvt.priors.acp_stru; the other fields are passed through unchanged.

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169
    """
    sig2 = np.asarray(sig2, dtype=float).ravel()
    stru = acp_stru(n, p, kappa, sig2, idx_ns)
    Vbeta = np.zeros((n * p + 1, n))
    for ii in range(n):
        for jj in range(n * p + 1):
            if ii == 0:
                Vbeta[jj, ii] = stru['Vbeta'][jj, ii]
            else:
                Vbeta[jj, ii] = stru['Vbeta'][jj, ii] + np.sum(
                    stru['Vbeta'][jj, :ii] + stru['beta0'][jj, :ii] ** 2 / sig2[:ii])
    return Struct(alp0=stru['alp0'], beta0=stru['beta0'], Valp=stru['Valp'], nu=stru['nu'],
                  S=stru['S'], Vbeta=Vbeta)
