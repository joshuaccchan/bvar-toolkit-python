import numpy as np
from scipy.special import gammaln


def ldiripdf(y, alpha):
    """bvt.ml.ldiripdf - log density of the Dirichlet distribution at each row of y.

      lden = bvt.ml.ldiripdf(y, alpha)

    y is N x k, each row a point in the simplex, or a k-vector for one point; alpha has
    k elements; lden is an N-vector, or a scalar for one point.

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    y = np.asarray(y, dtype=float)
    alpha = np.asarray(alpha, dtype=float).ravel()
    one = y.ndim == 1
    y = np.atleast_2d(y)
    if y.shape[1] != alpha.size:
        raise ValueError('dimensions do not match ')
    const = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))
    lden = const + np.log(y) @ (alpha - 1)
    return lden[0] if one else lden
