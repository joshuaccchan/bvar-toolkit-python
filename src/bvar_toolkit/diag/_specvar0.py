import numpy as np


def _columns(x):
    """Draws as an (R, k) matrix; any vector is one series."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1 or (x.ndim == 2 and 1 in x.shape):
        return x.reshape(-1, 1)
    return x


def specvar0(x, L):
    """bvt.diag.specvar0 - long-run variance of the sample mean of each column of a matrix of
    MCMC draws, from the spectral density at frequency zero estimated with the Bartlett window
    of Newey and West (1987):
      S = (gamma_0 + 2*sum_{l=1}^{L} (1 - l/(L+1))*gamma_l)/R,
    where gamma_l is the lag-l sample autocovariance with divisor R.

      S = bvt.diag.specvar0(x, L)

      x : R x k, one column per series (a vector is one series)
      L : truncation lag, a nonnegative integer below R
      S : k-vector

    For a textbook discussion, see Chan (2027), Section 6.5.

    See:
    Newey, W.K. and West, K.D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and
    Autocorrelation Consistent Covariance Matrix, Econometrica, 55(3): 703-708.
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 6.5.
    """
    x = _columns(x)
    R, k = x.shape
    if not (np.isscalar(L) and np.isreal(L) and L >= 0 and L == int(L) and L < R):
        raise ValueError(f'L must be a nonnegative integer below the number of draws ({R})')
    L = int(L)
    S = np.zeros(k)
    for j in range(k):
        xj = x[:, j] - np.mean(x[:, j])
        Sx = (xj @ xj) / R
        for ell in range(1, L + 1):
            w = 1 - ell / (L + 1)                   # Bartlett weight
            gamma = (xj[ell:] @ xj[:-ell]) / R
            Sx = Sx + 2 * w * gamma
        S[j] = Sx / R
    return S
