import numpy as np
from scipy.special import ndtr


def mixquantile(mu, sd, p):
    """bvt.forecast.mixquantile - quantiles of an equally weighted mixture of normals,
    such as a predictive density that averages one normal per posterior draw.

      q = bvt.forecast.mixquantile(mu, sd, p)

      mu, sd : one component per draw, the same length. bvt.forecast.simulate and
               bvt.forecast.predictive return exactly this pair for a variable at
               a horizon
      p      : probabilities in (0,1), any shape
      q      : the quantiles, the same shape as p

    The predictive probability of falling below x is mean(ndtr((x - mu)/sd)).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sd = np.asarray(sd, dtype=float).ravel()
    p = np.asarray(p, dtype=float)
    if mu.size != sd.size:
        raise ValueError('mu and sd must have the same length')
    if not np.all(sd > 0):
        raise ValueError('every component standard deviation must be positive')
    if not np.all((p > 0) & (p < 1)):
        raise ValueError('probabilities must lie strictly between 0 and 1')

    lo = np.min(mu - 12 * sd)
    hi = np.max(mu + 12 * sd)
    q = np.zeros(p.shape)
    for i, pi in enumerate(p.ravel()):
        a, b = lo, hi
        for _ in range(200):                        # bisection to machine tolerance
            m = (a + b) / 2
            if np.mean(ndtr((m - mu) / sd)) < pi:
                a = m
            else:
                b = m
            if b - a < 1e-12 * max(1, abs(m)):
                break
        q.flat[i] = (a + b) / 2
    return q
