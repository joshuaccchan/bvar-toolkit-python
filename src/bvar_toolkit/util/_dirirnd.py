import numpy as np

from .._rng import as_rng


def dirirnd(alp, N=1, rng=None):
    """bvt.util.dirirnd - draws from the Dirichlet distribution.

      draws = bvt.util.dirirnd(alp)
      draws = bvt.util.dirirnd(alp, N)
      draws = bvt.util.dirirnd(alp, N, rng)

    alp is a vector of r positive parameters. draws is N x r, one draw per row; N defaults to 1.
    rng is None, an integer seed, a numpy.random.Generator or a bvt.Stream.

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for time-varying parameter
    VARs with stochastic volatility, Journal of Applied Econometrics, 33(4), 509-532.
    """
    alp = np.asarray(alp, dtype=float).ravel()
    n = alp.size
    x = as_rng(rng).gamrnd(np.tile(alp, (N, 1)), 1.0, N, n)
    return x / x.sum(axis=1, keepdims=True)
