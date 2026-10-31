import numpy as np
from scipy.special import gammaln


def lgampdf(x, a, b):
    """bvt.ml.lgampdf - log density of the gamma distribution at x (shape a, RATE
    b; elementwise over array inputs).

      lden = bvt.ml.lgampdf(x, a, b)

    b is a rate. gamfit returns a scale, so a caller fitting with gamfit must
    invert it first (ckappa_hat = [a, 1/scale]).

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    x, a, b = np.asarray(x, dtype=float), np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lden = a * np.log(b) - gammaln(a) + (a - 1) * np.log(x) - b * x
    return lden[()] if lden.ndim == 0 else lden
