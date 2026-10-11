import numpy as np

from .._optim import fminbnd, fminsearch
from ..ml._acp import acp
from ._acp_redu import acp_redu
from ._acp_stru import acp_stru
from ._resid_var_ar4 import resid_var_ar4


def acp_opt_kappa(Y0, Y, Z, p, k0, form, idx_ns=None, symmetric=False, ridge=0.0):
    """bvt.priors.acp_opt_kappa - choose the two shrinkage hyperparameters of the asymmetric
    conjugate prior by maximizing the closed-form marginal likelihood.

      ml_opt, kappa_opt = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, k0, form)
      ml_opt, kappa_opt = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, k0, form, idx_ns)
      ...               = bvt.priors.acp_opt_kappa(..., symmetric=True)
      ...               = bvt.priors.acp_opt_kappa(..., ridge=1e-6)

      Y0, Y     : presample rows and the T x n estimation sample
      Z         : T x (n p + 1) lag matrix, intercept first (bvt.util.build_lags)
      p         : lag length
      k0        : starting values [kappa1, kappa2]; ignored when symmetric is True
      form      : 'redu' or 'stru', selecting bvt.priors.acp_redu or acp_stru
      idx_ns    : indices of nonstationary variables, counted from 0, whose first own lag gets
                  prior mean one (default none); no data transformation is implied
      symmetric : default False; True imposes kappa1 = kappa2 and searches (0,1) with fminbnd
      ridge     : passed to bvt.ml.acp at every evaluation; default 0. The SVAR-sign package's
                  copy of that likelihood adds 1e-6 to the diagonal of the posterior precision,
                  so reproducing that package's hyperparameters needs ridge=1e-6, which moves
                  the optimum
      ml_opt    : the maximized log marginal likelihood, bvt.ml.acp at kappa_opt
      kappa_opt : the full 4-vector [kappa1, kappa2, 1, 100] - kappa3 and kappa4 are held at
                  those values, as in the paper's application

    The asymmetric search runs fminsearch over log(kappa1) and log(kappa2). fminbnd and
    fminsearch are those of bvar_toolkit._optim.

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169.
    """
    if form not in ('redu', 'stru'):
        raise ValueError(f"form must be 'redu' or 'stru', got '{form}'")
    kappa3, kappa4 = 1, 100
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[1]
    sig2 = resid_var_ar4(Y0, Y)
    prior = acp_stru if form == 'stru' else acp_redu
    if symmetric:
        def f(k1):
            return -acp(p, Y, Z, prior(n, p, [k1, k1, kappa3, kappa4], sig2, idx_ns), ridge=ridge)
        kappa1, nml = fminbnd(f, 0, 1)
        return -nml, np.array([kappa1, kappa1, kappa3, kappa4], dtype=float)

    def f(k):
        return -acp(p, Y, Z, prior(n, p, [np.exp(k[0]), np.exp(k[1]), kappa3, kappa4], sig2, idx_ns),
                    ridge=ridge)
    k_opt, nml = fminsearch(f, np.log(np.asarray(k0, dtype=float)))
    return -nml, np.array([np.exp(k_opt[0]), np.exp(k_opt[1]), kappa3, kappa4], dtype=float)
