import numpy as np
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ._intlike_tvpsv import intlike_tvpsv
from ._mltvpsv_cvar import _batch_ml, _draws, _gauss_draws, _gauss_fit, _ig_draws, _ig_fit


def mltvpsv_tvpsv(Y, store_Sigtheta, store_Sigh, store_h0, store_theta0, prior, bigX, M,
                  rng=None):
    """bvt.ml.mltvpsv_tvpsv - log marginal likelihood of the TVP-SV model of Chan and
    Eisenstat (2018) by importance sampling. The importance density of each variance is
    inverse-gamma and that of each initial state Gaussian, fitted to the posterior draws;
    the states and log-volatilities are integrated out by bvt.ml.intlike_tvpsv. The M
    importance weights are split into 20 batches: lml is the mean of the 20 batch
    estimates of the log marginal likelihood, and lmlstd is their standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_tvpsv(Y, store_Sigtheta, store_Sigh, store_h0,
                                              store_theta0, prior, bigX, M)
      lml, lmlstd, out = bvt.ml.mltvpsv_tvpsv(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw
      prior  : log prior density, prior(Sigtheta, Sigh, theta0, h0)
      M      : importance draws, rounded up to a multiple of 20
      rng    : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out    : store_w, the M log weights; bigml, the 20 batch estimates

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    r = as_rng(rng)
    M = int(20 * np.ceil(M / 20))
    store_Sigtheta, store_Sigh = _draws(store_Sigtheta), _draws(store_Sigh)
    n = store_Sigh.shape[1]
    k = store_Sigtheta.shape[1]

    nuthetahat, Sthetahat = _ig_fit(store_Sigtheta)
    nuhhat, Shhat = _ig_fit(store_Sigh)

    theta0hat, theta0std, theta0pre = _gauss_fit(store_theta0)
    h0hat, h0std, h0pre = _gauss_fit(store_h0)

    big_Sigtheta = _ig_draws(nuthetahat, Sthetahat, M, r)
    big_Sigh = _ig_draws(nuhhat, Shhat, M, r)
    big_theta0 = _gauss_draws(theta0hat, theta0std, M, r)
    big_h0 = _gauss_draws(h0hat, h0std, M, r)

    store_w = np.zeros(M)
    cIS = nuthetahat @ np.log(Sthetahat) - np.sum(gammaln(nuthetahat)) \
        + nuhhat @ np.log(Shhat) - np.sum(gammaln(nuhhat)) \
        - .5 * (k + n) * np.log(2 * np.pi) - np.sum(np.log(np.diag(theta0std))) \
        - np.sum(np.log(np.diag(h0std)))

    def gIS(sthe, sh, a0, b0):
        da, db = a0 - theta0hat, b0 - h0hat
        return cIS - (nuthetahat + 1) @ np.log(sthe) - np.sum(Sthetahat / sthe) \
            - (nuhhat + 1) @ np.log(sh) - np.sum(Shhat / sh) \
            - .5 * da @ theta0pre @ da - .5 * db @ h0pre @ db

    for loop in range(M):
        Sigtheta = big_Sigtheta[loop]
        Sigh = big_Sigh[loop]
        theta0 = big_theta0[loop]
        h0 = big_h0[loop]

        llike = intlike_tvpsv(Y, Sigtheta, Sigh, bigX, h0, theta0, rng=r)[0]
        store_w[loop] = llike + prior(Sigtheta, Sigh, theta0, h0) \
            - gIS(Sigtheta, Sigh, theta0, h0)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)
