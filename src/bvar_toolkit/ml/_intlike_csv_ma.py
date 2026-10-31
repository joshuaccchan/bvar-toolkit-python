import numpy as np

from .._linalg import ma_solve
from .._rng import as_rng
from ._intlike_csv import _hish, _newton, _s2, _weights


def intlike_csv_ma(shortY, X, A, Sig, psi, rho, sigh2, R, rng=None):
    """bvt.ml.intlike_csv_ma - integrated likelihood of the BVAR-CSV-MA model at
    (A,Sig,psi,rho,sigh2): the common log-volatility path h is integrated out
    by importance sampling with R draws around the Newton-Raphson mode of its
    conditional density given the MA(1)-transformed residuals. The first
    observation carries the (1+psi^2) initialization variance both in the mode
    search and inside the deny_h likelihood evaluation.

      intlike, store_llike = bvt.ml.intlike_csv_ma(shortY, X, A, Sig, psi, rho, sigh2, R)
      intlike, store_llike = bvt.ml.intlike_csv_ma(..., rng)

    rng is None, an integer seed, a numpy.random.Generator or a bvt.Stream.

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    r = as_rng(rng)
    shortY = np.asarray(shortY, dtype=float)
    T, n = shortY.shape
    HiSH = _hish(rho, sigh2, T)

        # obtain the mode and negative Hessian of the conditional density of h
    CSig = np.linalg.cholesky(np.asarray(Sig, dtype=float))
    U = shortY - np.asarray(X, dtype=float) @ np.asarray(A, dtype=float)
    s2 = _s2(ma_solve(psi, U), CSig)
    s2m = s2.copy()
    s2m[0] = s2m[0] / (1 + psi**2)
    ht, Kh, CKh = _newton(s2m, n, HiSH, np.log(np.mean(s2m)) * np.ones(T))

        # evaluate the importance weights
    c = -T * n / 2 * np.log(2 * np.pi) - T * np.sum(np.log(np.diag(CSig))) \
        - n / 2 * np.log(1 + psi**2)

    def deny_h(h):
        return c - n / 2 * np.sum(h) \
            - .5 * (s2[0] / ((1 + psi**2) * np.exp(h[0])) + np.sum(s2[1:] / np.exp(h[1:])))

    return _weights(ht, Kh, CKh, HiSH, rho, sigh2, int(R), r, deny_h)
