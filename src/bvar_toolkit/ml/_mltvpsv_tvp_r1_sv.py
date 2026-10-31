import numpy as np
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ._intlike_tvpsv import intlike_tvpsv
from ._mltvpsv_cvar import (_batch_ml, _draws, _gauss_draws, _gauss_fit, _ig_draws, _ig_fit,
                            _regressors)


def mltvpsv_tvp_r1_sv(Y, store_beta, store_Siggam, store_Sigh, store_h0, store_gam0, prior,
                      Xtilde, W, M, rng=None):
    """bvt.ml.mltvpsv_tvp_r1_sv - log marginal likelihood of the TVP-R1-SV model of Chan and
    Eisenstat (2018) by importance sampling. The importance density of each variance is
    inverse-gamma and that of beta and of each initial state Gaussian, fitted to the
    posterior draws; the impact states gam_t and the log-volatilities are integrated out
    by bvt.ml.intlike_tvpsv, applied to Y - Xtilde @ beta with W as the regressors. The M
    importance weights are split into 20 batches: lml is the mean of the 20 batch
    estimates of the log marginal likelihood, and lmlstd is their standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_tvp_r1_sv(Y, store_beta, store_Siggam, store_Sigh,
                                                  store_h0, store_gam0, prior, Xtilde, W, M)
      lml, lmlstd, out = bvt.ml.mltvpsv_tvp_r1_sv(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw
      prior  : log prior density, prior(beta, Siggam, Sigh, gam0, h0)
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
    Y = np.asarray(Y, dtype=float).ravel()
    store_Siggam, store_Sigh = _draws(store_Siggam), _draws(store_Sigh)
    n = store_Sigh.shape[1]
    kgam = store_Siggam.shape[1]
    Xtilde = _regressors(Xtilde, Y.size)

    nugamhat, Sgamhat = _ig_fit(store_Siggam)
    nuhhat, Shhat = _ig_fit(store_Sigh)

    betahat, betastd, betapre = _gauss_fit(store_beta)
    gam0hat, gam0std, gam0pre = _gauss_fit(store_gam0)
    h0hat, h0std, h0pre = _gauss_fit(store_h0)
    kbeta = betahat.size

    big_Siggam = _ig_draws(nugamhat, Sgamhat, M, r)
    big_Sigh = _ig_draws(nuhhat, Shhat, M, r)
    big_beta = _gauss_draws(betahat, betastd, M, r)
    big_gam0 = _gauss_draws(gam0hat, gam0std, M, r)
    big_h0 = _gauss_draws(h0hat, h0std, M, r)

    store_w = np.zeros(M)
    cIS = -.5 * kbeta * np.log(2 * np.pi) - np.sum(np.log(np.diag(betastd))) \
        + nugamhat @ np.log(Sgamhat) - np.sum(gammaln(nugamhat)) \
        + nuhhat @ np.log(Shhat) - np.sum(gammaln(nuhhat)) \
        - .5 * (kgam + n) * np.log(2 * np.pi) - np.sum(np.log(np.diag(gam0std))) \
        - np.sum(np.log(np.diag(h0std)))

    def gIS(b, sg, sh, a0, b0):
        d, da, db = b - betahat, a0 - gam0hat, b0 - h0hat
        return cIS - .5 * d @ betapre @ d \
            - (nugamhat + 1) @ np.log(sg) - np.sum(Sgamhat / sg) \
            - (nuhhat + 1) @ np.log(sh) - np.sum(Shhat / sh) \
            - .5 * da @ gam0pre @ da - .5 * db @ h0pre @ db

    for loop in range(M):
        beta = big_beta[loop]
        Siggam = big_Siggam[loop]
        Sigh = big_Sigh[loop]
        gam0 = big_gam0[loop]
        h0 = big_h0[loop]

        llike = intlike_tvpsv(Y - Xtilde @ beta, Siggam, Sigh, W, h0, gam0, rng=r)[0]
        store_w[loop] = llike + prior(beta, Siggam, Sigh, gam0, h0) \
            - gIS(beta, Siggam, Sigh, gam0, h0)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)
