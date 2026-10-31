import numpy as np
import scipy.sparse as sp

from .._linalg import BandChol
from .._rng import as_rng
from ._intlike_tvp import _rw_prec


def _adaptive(R, draw):
    """R log weights, or with R None ten of them, extended to 10*(floor(v) + 1) when the
    variance v of their mean exceeds 1."""
    if R is not None:
        return np.array([draw() for _ in range(int(R))])
    R = 10
    store_llike = np.array([draw() for _ in range(R)])
        # increase simulation size if the variance of the log-likelihood > 1
    var_llike = np.var(store_llike, ddof=1) / R
    if var_llike > 1:
        RR = int(np.floor(var_llike))
        store_llike = np.r_[store_llike, [draw() for _ in range(R * RR)]]
    return store_llike


def intlike_cvarsv(Y, theta, Sigh, bigX, h0, R=None, rng=None):
    """bvt.ml.intlike_cvarsv - log integrated likelihood of a constant-coefficient VAR
    with random-walk stochastic volatility, given the coefficients, by importance
    sampling of the log-volatilities h. The importance density of h is Gaussian at the
    mode of its conditional density, found by Newton-Raphson, with precision from the
    negative Hessian there.

      llike = bvt.ml.intlike_cvarsv(Y, theta, Sigh, bigX, h0)
      llike = bvt.ml.intlike_cvarsv(..., R, rng)

      Y     : T*n data, stacked by period
      theta : the coefficients, bigX @ theta the conditional mean of Y
      Sigh  : n variances of the log-volatility innovations
      h0    : n initial log-volatilities
      R     : number of importance draws. None, R = 10, and when the variance of
              the mean log weight exceeds 1 the draws are increased to
              R*(floor of that variance + 1).
      rng   : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    The Newton iteration stops when no log-volatility moves by more than .01, or
    after 100 steps, when the importance density is centered at h0 instead.

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()
    Sigh = np.asarray(Sigh, dtype=float).ravel()
    h0 = np.asarray(h0, dtype=float).ravel()
    max_loop = 100
    n = Sigh.size
    T = Y.size // n
        # obtain the proposal density
    HinvSH_h = _rw_prec(1 / Sigh, T, n)
    alph = np.tile(h0, T)
    s2 = (Y - bigX @ theta)**2
    e_h = 1.0
    ht = np.tile(h0, T)
    count = 0
    while e_h > .01 and count < max_loop:
        einvhts2 = np.exp(-ht) * s2
        gh = -(HinvSH_h @ (ht - alph)) - .5 * (1 - einvhts2)
        Kh = (HinvSH_h + .5 * sp.diags_array(einvhts2)).tocsc()
        CKh = BandChol(Kh)
        newht = ht + CKh.solve_t(CKh.solve(gh))
        e_h = np.max(np.abs(newht - ht))
        ht = newht
        count += 1
    if count == max_loop:
        ht = np.tile(h0, T)
        einvhts2 = np.exp(-ht) * s2
        Kh = (HinvSH_h + .5 * sp.diags_array(einvhts2)).tocsc()
        CKh = BandChol(Kh)
        # Kh's factor CKh is that of the last Newton step, or of the fallback

        # evaluate the importance weights
    c_pri = -T * n / 2 * np.log(2 * np.pi) - .5 * T * np.sum(np.log(Sigh))
    c_IS = -T * n / 2 * np.log(2 * np.pi) + CKh.logdet() / 2
    e = Y - bigX @ theta

    def draw():
        hc = ht + CKh.solve_t(r.randn(T * n))
        llike = -T * n / 2 * np.log(2 * np.pi) - .5 * np.sum(hc) - .5 * e @ (np.exp(-hc) * e)
        d0, d1 = hc - alph, hc - ht
        return llike + (c_pri - .5 * d0 @ (HinvSH_h @ d0)) - (c_IS - .5 * d1 @ (Kh @ d1))

    store_llike = _adaptive(R, draw)
    maxllike = np.max(store_llike)
    return np.log(np.mean(np.exp(store_llike - maxllike))) + maxllike
