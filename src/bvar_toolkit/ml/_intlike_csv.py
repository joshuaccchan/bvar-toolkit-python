import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

from .._linalg import BandChol
from .._rng import as_rng


def _hish(rho, sigh2, T):
    """The prior precision Hrho' diag((1-rho^2)/sigh2, 1/sigh2, ...) Hrho of the AR(1) path."""
    Hrho = (sp.eye_array(T, format='csc') - rho * sp.eye_array(T, k=-1, format='csc')).tocsc()
    D = sp.diags_array(np.r_[(1 - rho**2) / sigh2, 1 / sigh2 * np.ones(T - 1)])
    return (Hrho.T @ D @ Hrho).tocsc()


def _newton(s2, n, HiSH, ht):
    """The Newton-Raphson mode of the conditional density of h from ht, with Kh and its
    factor at the last step."""
    errh = 1.0
    while errh > 10**(-3):
        eht = np.exp(ht)
        sieht = s2 / eht
        fh = -n / 2 + .5 * sieht
        Gh = .5 * sieht
        Kh = (HiSH + sp.diags_array(Gh)).tocsc()
        CKh = BandChol(Kh)
        newht = CKh.solve_t(CKh.solve(fh + Gh * ht))
        errh = np.max(np.abs(newht - ht))
        ht = newht
    return ht, Kh, CKh


def _em_t(s2_0, n, nu, HiSH, ht):
    """The mode of the conditional density of h under Student-t errors, by EM with an inner
    Newton-Raphson step on h; s2_0 are the squared standardized errors."""
    errh_out = 1.0
    while errh_out > 10**(-3):
            # E-step
        tmps2 = s2_0 / np.exp(ht)
        Eilam = (nu + n) / (tmps2 + nu)
        s2 = s2_0 * Eilam
            # M-step
        htt = ht
        errh_in = 1.0
        while errh_in > 10**(-3):
            eht = np.exp(htt)
            sieht = s2 / eht
            fh = -n / 2 + .5 * sieht
            Gh = .5 * sieht
            Kh = (HiSH + sp.diags_array(Gh)).tocsc()
            CKh = BandChol(Kh)
            newht = CKh.solve_t(CKh.solve(fh + Gh * htt))
            errh_in = np.max(np.abs(newht - htt))
            htt = newht
        errh_out = np.max(np.abs(ht - htt))
        ht = htt
        # compute negative Hessian
    Gh = (nu + n) / (2 * nu) * (s2_0 * np.exp(ht)) / ((np.exp(ht) + s2_0 / nu)**2)
    Kh = (HiSH + sp.diags_array(Gh)).tocsc()
    return ht, Kh, BandChol(Kh)


def _weights(ht, Kh, CKh, HiSH, rho, sigh2, R, r, deny):
    """The log importance weights of R draws of h from N(ht, inv(Kh)), and the log of their
    mean."""
    T = ht.size
    c_pri = -T / 2 * np.log(2 * np.pi * sigh2) - .5 * np.log(1 / (1 - rho**2))
    c_IS = -T / 2 * np.log(2 * np.pi) + CKh.logdet() / 2
    store_llike = np.zeros(R)
    for i in range(R):
        hc = ht + CKh.solve_t(r.randn(T))
        d = hc - ht
        store_llike[i] = deny(hc) + (c_pri - .5 * hc @ (HiSH @ hc)) \
            - (c_IS - .5 * d @ (Kh @ d))
    maxllike = np.max(store_llike)
    return np.log(np.mean(np.exp(store_llike - maxllike))) + maxllike, store_llike


def _s2(U, CSig):
    """Row sums of squares of U @ inv(CSig).T."""
    return np.sum(solve_triangular(CSig, U.T, lower=True)**2, axis=0)


def intlike_csv(shortY, X, A, Sig, rho, sigh2, R, rng=None):
    """bvt.ml.intlike_csv - integrated (observed-data) likelihood of the BVAR-CSV
    model at (A,Sig,rho,sigh2): the common log-volatility path h is integrated
    out by importance sampling with R draws from a Gaussian approximation
    N(ht, inv(Kh)) built at the Newton-Raphson mode of the conditional density
    of h.

      intlike, store_llike = bvt.ml.intlike_csv(shortY, X, A, Sig, rho, sigh2, R)
      intlike, store_llike = bvt.ml.intlike_csv(..., rng)

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
    ht, Kh, CKh = _newton(s2, n, HiSH, np.log(np.mean(s2)) * np.ones(T))

        # evaluate the importance weights
    c = -T * n / 2 * np.log(2 * np.pi) - T * np.sum(np.log(np.diag(CSig)))

    def deny_h(h):
        return c - n / 2 * np.sum(h) - .5 * np.sum(s2 / np.exp(h))

    return _weights(ht, Kh, CKh, HiSH, rho, sigh2, int(R), r, deny_h)
