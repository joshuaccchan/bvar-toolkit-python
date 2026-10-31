import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

from .._linalg import BandChol
from .._rng import as_rng
from ._intlike_cvarsv import _adaptive
from ._intlike_tvp import _rw_prec


def intlike_tvpsv(Y, Sigtheta, Sigh, bigX, h0, theta0, R=None, rng=None):
    """bvt.ml.intlike_tvpsv - log integrated likelihood of a TVP regression with
    random-walk stochastic volatility, by importance sampling of the log-volatilities
    h. The states theta_t are integrated out analytically for each draw of h; the
    importance density of h is Gaussian at the mode of its marginal density, found by
    an EM algorithm, with precision from the negative Hessian there.

      llike, store_llike = bvt.ml.intlike_tvpsv(Y, Sigtheta, Sigh, bigX, h0, theta0)
      llike, store_llike = bvt.ml.intlike_tvpsv(..., R, rng)

      Y        : T*n data, stacked by period
      Sigtheta : k variances of the state innovations
      Sigh     : n variances of the log-volatility innovations
      bigX     : T*n x T*k regressor matrix of the stacked states
      h0, theta0 : n and k initial states
      R        : number of importance draws. None, R = 10, and when the variance
                 of the mean log weight exceeds 1 the draws are increased to
                 R*(floor of that variance + 1).
      rng      : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      store_llike : the log importance weights

    The EM algorithm stops when no log-volatility moves by more than .1, or after 100
    iterations; its Newton step stops at the same tolerance or after 1000 steps. If
    the EM reaches 100 iterations, the importance density is centered at h0 instead.

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float).ravel()
    Sigtheta = np.asarray(Sigtheta, dtype=float).ravel()
    Sigh = np.asarray(Sigh, dtype=float).ravel()
    h0 = np.asarray(h0, dtype=float).ravel()
    theta0 = np.asarray(theta0, dtype=float).ravel()
    bigX = sp.csc_array(bigX)
    n = Sigh.size
    T = Y.size // n
    k = Sigtheta.size
        # obtain the mode of the marginal density of h (unconditional on theta)
    HinvSH_theta = _rw_prec(1 / Sigtheta, T, k)
    alptheta = np.tile(theta0, T)
    HinvSH_h = _rw_prec(1 / Sigh, T, n)
    alph = np.tile(h0, T)
    bigXt = bigX.T.toarray()

    def estep(h):
        XinvSig = bigX.T @ sp.diags_array(np.exp(-h))
        CK = BandChol((HinvSH_theta + XinvSig @ bigX).tocsc())
        thetahat = CK.solve_t(CK.solve(XinvSig @ Y + HinvSH_theta @ alptheta))
        zhat = np.sum(CK.solve(bigXt)**2, axis=0) + (Y - bigX @ thetahat)**2
        return XinvSig, CK, zhat

    e_h = 1.0
    ht = np.tile(h0, T)
    countout = 0
    while e_h > .1 and countout < 100:
            # E-step
        XinvSig, CK, zhat = estep(ht)
            # M-step
        e_hj = 1.0
        htt = ht
        countin = 0
        while e_hj > .1 and countin < 1000:
            einvhttzhat = np.exp(-htt) * zhat
            gQ = -(HinvSH_h @ (htt - alph)) - .5 * (1 - einvhttzhat)
            negHQ = (HinvSH_h + .5 * sp.diags_array(einvhttzhat)).tocsc()
            CQ = BandChol(negHQ)
            newhtt = htt + CQ.solve_t(CQ.solve(gQ))
            e_hj = np.max(np.abs(newhtt - htt))
            htt = newhtt
            countin += 1
        if countin < 1000:
            e_h = np.max(np.abs(ht - htt))
            ht = htt
        countout += 1

    if countout == 100:
        ht = np.tile(h0, T)
        XinvSig, CK, zhat = estep(ht)
        negHQ = (HinvSH_h + .5 * sp.diags_array(np.exp(-ht) * zhat)).tocsc()
    Z = np.asarray(XinvSig.T @ CK.solve_t(CK.solve(bigXt)))
    HH = -.5 * Z.T * (np.eye(n * T) - Z)
    Kh = negHQ.toarray() - HH
    Cg = np.linalg.cholesky(Kh)

        # evaluate the importance weights
    c_pri = -T * n / 2 * np.log(2 * np.pi) - .5 * T * np.sum(np.log(Sigh))
    c_IS = -T * n / 2 * np.log(2 * np.pi) + np.sum(np.log(np.diag(Cg)))
    cdeny = -T * n / 2 * np.log(2 * np.pi) - T / 2 * np.sum(np.log(Sigtheta))
    aHa = alptheta @ (HinvSH_theta @ alptheta)

    def deny_h(h):
            # the log likelihood given the log-volatilities h, the states integrated out
        iSig = np.exp(-h)
        XinvSig_h = bigX.T @ sp.diags_array(iSig)
        CKd = BandChol((HinvSH_theta + XinvSig_h @ bigX).tocsc())
        dtheta = XinvSig_h @ Y + HinvSH_theta @ alptheta
        return cdeny - .5 * np.sum(h) - CKd.logdet() / 2 \
            - .5 * (Y @ (iSig * Y) + aHa - dtheta @ CKd.solve_t(CKd.solve(dtheta)))

    def draw():
        hc = ht + solve_triangular(Cg, r.randn(T * n), lower=True, trans='T')
        d0, d1 = hc - alph, hc - ht
        return deny_h(hc) + (c_pri - .5 * d0 @ (HinvSH_h @ d0)) - (c_IS - .5 * d1 @ (Kh @ d1))

    store_llike = _adaptive(R, draw)
    maxllike = np.max(store_llike)
    return np.log(np.mean(np.exp(store_llike - maxllike))) + maxllike, store_llike
