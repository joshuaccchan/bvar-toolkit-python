import numpy as np
from scipy.special import gammaln

from .._rng import as_rng
from ._intlike_csv import _em_t, _hish, _s2, _weights


def _deny_t(s2, CSig, n, nu):
    """The log likelihood of h under Student-t errors with squared standardized errors s2."""
    T = s2.size
    c = T * (gammaln((nu + n) / 2) - gammaln(nu / 2) - n / 2 * np.log(nu * np.pi)) \
        - T * np.sum(np.log(np.diag(CSig)))

    def deny_h(h):
        return c - n / 2 * np.sum(h) - (nu + n) / 2 * np.sum(np.log(1 + (s2 / np.exp(h)) / nu))
    return deny_h


def intlike_t_csv(shortY, X, A, Sig, rho, sigh2, nu, R, rng=None):
    """bvt.ml.intlike_t_csv - integrated likelihood of the BVAR-t-CSV model at
    (A,Sig,rho,sigh2,nu): both the Student-t mixture scales lam (analytically,
    per-observation t densities) and the common log-volatility path h (by
    importance sampling, R draws) are integrated out. The mode search is an
    EM-within-Newton scheme (E-step over lam, inner NR over h) and the IS
    covariance uses the exact negative Hessian of the t-marginal conditional.

      intlike, store_llike = bvt.ml.intlike_t_csv(shortY, X, A, Sig, rho, sigh2, nu, R)
      intlike, store_llike = bvt.ml.intlike_t_csv(..., rng)

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
    s2 = _s2(U, CSig)
    ht, Kh, CKh = _em_t(s2, n, nu, HiSH, np.log(np.mean(s2)) * np.ones(T))

        # evaluate the importance weights
    return _weights(ht, Kh, CKh, HiSH, rho, sigh2, int(R), r, _deny_t(s2, CSig, n, nu))
