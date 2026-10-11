import numpy as np
from scipy.special import gammaln


def mgammaln(n, x):
    """bvt.util.mgammaln - log of the multivariate gamma function Gamma_n(x).

      k = bvt.util.mgammaln(n, x)

      n : dimension, a positive integer
      x : scalar argument; needs x > (n-1)/2, else scipy's gammaln returns inf, or log|Gamma| of
          a negative argument
      k : log Gamma_n(x)

    See:
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large Bayesian
    VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    return n * (n - 1) / 4 * np.log(np.pi) + np.sum(gammaln(x - 0.5 * np.arange(n)))
