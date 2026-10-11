import numpy as np
from scipy.special import ndtr, ndtri

from .._rng import as_rng


def tnormrnd(mu, sigma2, a, b, N=None, rng=None):
    """bvt.util.tnormrnd - N draws from the normal truncated to (a, b), by inverse transform.

      t = bvt.util.tnormrnd(mu, sigma2, a, b)
      t = bvt.util.tnormrnd(mu, sigma2, a, b, N)
      t = bvt.util.tnormrnd(mu, sigma2, a, b, N, rng)

      mu, sigma2 : mean and VARIANCE of the untruncated normal, either scalars, which are
                   expanded to N identical values, or length-N vectors
      a, b       : truncation bounds, scalar or length-N
      N          : number of draws (default len(mu))
      rng        : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      t          : N-vector of draws

    See:
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large Bayesian
    VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float)).ravel()
    sigma2 = np.atleast_1d(np.asarray(sigma2, dtype=float)).ravel()
    K = mu.size
    N = K if N is None else int(N)
    if (K != N or sigma2.size != N) and K != 1:
        raise ValueError('dimensions of mu and sigma must equal N')
    if K == 1:
        mu = np.full(N, mu[0])
        sigma2 = np.full(N, sigma2[0]) if sigma2.size == 1 else sigma2
    sigma = np.sqrt(sigma2)
    u = as_rng(rng).rand(N)
    p1 = ndtr((np.asarray(a, dtype=float) - mu) / sigma)
    p2 = ndtr((np.asarray(b, dtype=float) - mu) / sigma)
    return mu + sigma * ndtri(p1 + (p2 - p1) * u)
