import numpy as np
from scipy.special import gammaln


def linvgammpdf(y, a, b):
    """bvt.ml.linvgammpdf - log density of the inverse-gamma IG(a,b) distribution
    at y (shape a, scale b; elementwise over array inputs).

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y, a, b))
    lden = a * np.log(b) - gammaln(a) - (a + 1) * np.log(y) - b / y
    return lden[()] if np.ndim(lden) == 0 else lden
