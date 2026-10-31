import numpy as np
from scipy.linalg import cholesky_banded
from scipy.linalg.lapack import dtbtrs

from .._rng import as_rng


def csv_armh(s2, rho, sigh2, h, n, forced_accept=False, ht_start=None, c_reject=3,
             max_iter_mode=500, max_iter_ar=1000, rng=None):
    """bvt.sv.csv_armh - one accept-reject Metropolis-Hastings sweep for the common
    stochastic volatility log-volatility path h, marginal of the mixture indicators.

      h, is_accept = bvt.sv.csv_armh(s2, rho, sigh2, h, n)
      h, is_accept = bvt.sv.csv_armh(..., forced_accept, ht_start)
      h, is_accept = bvt.sv.csv_armh(..., c_reject=3)

      s2            : T-vector, sum over the n series of squared (orthogonalized,
                      lambda-scaled) errors at each t
      rho, sigh2    : AR(1) coefficient and innovation variance of h
      h, n          : current log-volatility path (T-vector), and number of series
      forced_accept : take the proposal regardless of the MH ratio; default False
      ht_start      : starting point of the mode search; default h
      is_accept     : 1 if the MH proposal was accepted
      c_reject      : c in the envelope target <= c * proposal; default 3, an
                      efficiency setting that leaves the target unchanged
      max_iter_mode, max_iter_ar : caps on the two loops; defaults 500 and 1000
      rng           : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    An independence sampler, so a bad starting h can leave the chain stuck there
    silently; a rejected candidate returns h unchanged, so check is_accept.

    See:
    Chan, J.C.C. (2017). The stochastic volatility in mean model with time-varying
    parameters: an application to inflation modeling, Journal of Business and
    Economic Statistics, 35(1), 17-28, for the accept-reject step this adapts.
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    tol_mode = 1e-3                       # fixed, not an option
    if not (np.ndim(forced_accept) == 0 and isinstance(forced_accept, (bool, np.bool_, int,
                                                                       float, np.number))):
        raise ValueError(f'forced_accept must be a logical scalar; got {forced_accept!r}. To '
                         'supply ht_start while leaving the flag alone, call '
                         'csv_armh(..., False, ht_start).')
    if not (np.ndim(c_reject) == 0 and np.isrealobj(c_reject) and np.isfinite(c_reject)
            and c_reject > 0):
        raise ValueError(f'c_reject must be a finite positive real scalar; got {c_reject!r}. '
                         'Zero freezes the chain and a negative value silently acts as its '
                         'magnitude.')
    max_iter_mode = _check_cap(max_iter_mode, 'max_iter_mode')
    max_iter_ar = _check_cap(max_iter_ar, 'max_iter_ar')
    r = as_rng(rng)
    s2 = np.asarray(s2, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    ht = h.copy() if ht_start is None or np.size(ht_start) == 0 else \
        np.asarray(ht_start, dtype=float).ravel()

    is_accept = 0
    T = s2.size
    # HiSH = Hrho' diag(dv) Hrho, tridiagonal with diagonal d and off-diagonal e
    dv = np.r_[(1 - rho**2) / sigh2, 1 / sigh2 * np.ones(T - 1)]
    d = dv.copy()
    d[:-1] = dv[:-1] + (-rho * dv[1:]) * (-rho)
    e = -rho * dv[1:]
    errh = np.inf
    it_mode = 0
    while not errh <= tol_mode:           # NaN fails this: chol or the cap then raises
        it_mode += 1
        if it_mode > max_iter_mode:
            raise RuntimeError(f'mode search did not converge to {tol_mode:g} in {max_iter_mode} '
                               f'iterations (last max|dh| = {errh:g})')
        sieht = s2 / np.exp(ht)
        fh = -n / 2 + .5 * sieht
        Gh = .5 * sieht
        Kd = d + Gh                       # Kh = HiSH + diag(Gh)
        CKh = cholesky_banded(_band(Kd, e), lower=True)
        v = dtbtrs(CKh, (fh + Gh * ht)[:, None], uplo='L')[0]
        newht = dtbtrs(CKh, v, uplo='L', trans='T')[0][:, 0]
        errh = np.max(np.abs(newht - ht))
        ht = newht
    # AR-step; Kh and its factor are those of the last iteration of the mode search
    logc = _logtarget(ht, d, e, n, s2) + np.log(c_reject)
    flag = 0
    it_ar = 0
    while flag == 0:
        it_ar += 1
        if it_ar > max_iter_ar:
            raise RuntimeError(f'no accept-reject proposal accepted in {max_iter_ar} draws at '
                               f'c_reject = {c_reject:g}; lower c_reject, or check s2, rho and '
                               'sigh2')
        hc = ht + dtbtrs(CKh, r.randn(T)[:, None], uplo='L', trans='T')[0][:, 0]
        alpARc = _logtarget(hc, d, e, n, s2) + .5 * _quad(hc - ht, Kd, e) - logc
        if alpARc > np.log(r.rand()):
            flag = 1
    # MH-step
    alpAR = _logtarget(h, d, e, n, s2) + .5 * _quad(h - ht, Kd, e) - logc
    if alpAR < 0:
        alpMH = 1.0
    elif alpARc < 0:
        alpMH = -alpAR
    else:
        alpMH = alpARc - alpAR
    if alpMH > np.log(r.rand()) or forced_accept:
        h = hc
        is_accept = 1
    return h, is_accept


def _band(d, e):
    return np.vstack([d, np.r_[e, 0.0]]) if d.size > 1 else d[None, :]


def _tri(d, e, x):
    y = d * x
    y[:-1] += e * x[1:]
    y[1:] += e * x[:-1]
    return y


def _logtarget(x, d, e, n, s2):
    return -.5 * x @ _tri(d, e, x) - n / 2 * np.sum(x) - .5 * np.exp(-x) @ s2


def _quad(v, Kd, e):
    return v @ _tri(Kd, e, v)


def _check_cap(v, name):
    if not (np.ndim(v) == 0 and np.isrealobj(v) and np.isfinite(v) and v >= 1 and v == int(v)):
        raise ValueError(f'{name} must be a finite positive integer; got {v!r}. Inf would '
                         'restore the unbounded loop this argument exists to prevent.')
    return int(v)
