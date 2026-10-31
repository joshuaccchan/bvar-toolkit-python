import numpy as np
from scipy.special import ndtr


def ltnormpdf(x, mu, sig2, lb, ub):
    """bvt.ml.ltnormpdf - log density of the normal N(mu,sig2) truncated to (lb,ub);
    the density counterpart of bvt.util.tnormrnd.

      lden = bvt.ml.ltnormpdf(x, mu, sig2, lb, ub)

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    x = np.asarray(x, dtype=float)
    c = -.5 * np.log(2 * np.pi * sig2) - np.log(ndtr((ub - mu) / np.sqrt(sig2))
                                               - ndtr((lb - mu) / np.sqrt(sig2)))
    lden = c - .5 / sig2 * (x - mu)**2
    return lden[()] if np.ndim(lden) == 0 else lden
