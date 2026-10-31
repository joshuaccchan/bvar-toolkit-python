import numpy as np

from .._rng import as_rng


def igrnd(nu, S, rng=None):
    """bvt.util.igrnd - draws from the inverse-gamma distribution in the (nu, S)
    parameterization used throughout this toolkit and the papers:

          x ~ IG(nu, S)   with density proportional to x^(-nu-1) exp(-S/x),
          mean S/(nu-1) for nu > 1.

      x = bvt.util.igrnd(nu, S)
      x = bvt.util.igrnd(nu, S, rng)

    nu and S must be positive, and may be scalars or conformable arrays; the result has the size
    of the expansion, one independent draw per element. rng is None, an integer seed, a
    numpy.random.Generator or a bvt.Stream.

    The conjugate variance step of a Gaussian model - with prior IG(nu0, S0) and residuals e -
    is

          sig2 = bvt.util.igrnd(nu0 + T/2, S0 + np.sum(e**2)/2)
    """
    nu = np.asarray(nu, dtype=float)
    S = np.asarray(S, dtype=float)
    if np.any(nu <= 0) or np.any(S <= 0):
        raise ValueError('nu and S must be positive')
    x = 1.0 / as_rng(rng).gamrnd(nu, 1.0 / S)
    return x[()] if np.ndim(x) == 0 else x
