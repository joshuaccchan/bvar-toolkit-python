import numpy as np
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ._intlike_tvpsv import intlike_tvpsv
from ._mltvpsv_cvar import (_batch_ml, _draws, _gauss_draws, _gauss_fit, _ig_draws, _ig_fit,
                            _regressors)


def mltvpsv_tvp_r2_sv(Y, store_gam, store_Sigbeta, store_Sigh, store_h0, store_beta0, prior,
                      Xtilde, W, M, rng=None):
    """bvt.ml.mltvpsv_tvp_r2_sv - log marginal likelihood of the TVP-R2-SV model of Chan and
    Eisenstat (2018) by importance sampling. The importance density of each variance is
    inverse-gamma and that of gam and of each initial state Gaussian, fitted to the
    posterior draws; the coefficient states beta_t and the log-volatilities are integrated
    out by bvt.ml.intlike_tvpsv, applied to Y - W @ gam with Xtilde as the regressors. The
    M importance weights are split into 20 batches: lml is the mean of the 20 batch
    estimates of the log marginal likelihood, and lmlstd is their standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_tvp_r2_sv(Y, store_gam, store_Sigbeta, store_Sigh,
                                                  store_h0, store_beta0, prior, Xtilde, W, M)
      lml, lmlstd, out = bvt.ml.mltvpsv_tvp_r2_sv(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw
      prior  : log prior density, prior(gam, Sigbeta, Sigh, beta0, h0)
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
    store_Sigbeta, store_Sigh = _draws(store_Sigbeta), _draws(store_Sigh)
    n = store_Sigh.shape[1]
    kbeta = store_Sigbeta.shape[1]
    W = _regressors(W, Y.size)

    nubetahat, Sbetahat = _ig_fit(store_Sigbeta)
    nuhhat, Shhat = _ig_fit(store_Sigh)

    gamhat, gamstd, gampre = _gauss_fit(store_gam)
    beta0hat, beta0std, beta0pre = _gauss_fit(store_beta0)
    h0hat, h0std, h0pre = _gauss_fit(store_h0)
    kgam = gamhat.size

    big_Sigbeta = _ig_draws(nubetahat, Sbetahat, M, r)
    big_Sigh = _ig_draws(nuhhat, Shhat, M, r)
    big_gam = _gauss_draws(gamhat, gamstd, M, r)
    big_beta0 = _gauss_draws(beta0hat, beta0std, M, r)
    big_h0 = _gauss_draws(h0hat, h0std, M, r)

    store_w = np.zeros(M)
    cIS = -.5 * kgam * np.log(2 * np.pi) - np.sum(np.log(np.diag(gamstd))) \
        + nubetahat @ np.log(Sbetahat) - np.sum(gammaln(nubetahat)) \
        + nuhhat @ np.log(Shhat) - np.sum(gammaln(nuhhat)) \
        - .5 * (kbeta + n) * np.log(2 * np.pi) - np.sum(np.log(np.diag(beta0std))) \
        - np.sum(np.log(np.diag(h0std)))

    def gIS(g, sb, sh, a0, b0):
        d, da, db = g - gamhat, a0 - beta0hat, b0 - h0hat
        return cIS - .5 * d @ gampre @ d \
            - (nubetahat + 1) @ np.log(sb) - np.sum(Sbetahat / sb) \
            - (nuhhat + 1) @ np.log(sh) - np.sum(Shhat / sh) \
            - .5 * da @ beta0pre @ da - .5 * db @ h0pre @ db

    for loop in range(M):
        gam = big_gam[loop]
        Sigbeta = big_Sigbeta[loop]
        Sigh = big_Sigh[loop]
        beta0 = big_beta0[loop]
        h0 = big_h0[loop]

        llike = intlike_tvpsv(Y - W @ gam, Sigbeta, Sigh, Xtilde, h0, beta0, rng=r)[0]
        store_w[loop] = llike + prior(gam, Sigbeta, Sigh, beta0, h0) \
            - gIS(gam, Sigbeta, Sigh, beta0, h0)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)
