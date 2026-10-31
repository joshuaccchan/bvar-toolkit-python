import numpy as np
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ._intlike_cvarsv import intlike_cvarsv
from ._mltvpsv_cvar import _batch_ml, _draws, _gauss_draws, _gauss_fit, _ig_draws, _ig_fit


def mltvpsv_cvarsv(Y, store_theta, store_Sigh, store_h0, prior, bigX, M, rng=None):
    """bvt.ml.mltvpsv_cvarsv - log marginal likelihood of the CVAR-SV model of Chan and
    Eisenstat (2018) by importance sampling. The importance density of each variance is
    inverse-gamma and that of the coefficients and of the initial log-volatilities
    Gaussian, fitted to the posterior draws; the log-volatilities are integrated out by
    bvt.ml.intlike_cvarsv. The M importance weights are split into 20 batches: lml is the
    mean of the 20 batch estimates of the log marginal likelihood, and lmlstd is their
    standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_cvarsv(Y, store_theta, store_Sigh, store_h0, prior,
                                               bigX, M)
      lml, lmlstd, out = bvt.ml.mltvpsv_cvarsv(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw
      prior  : log prior density, prior(theta, Sigh, h0)
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
    store_theta, store_Sigh = _draws(store_theta), _draws(store_Sigh)
    n = store_Sigh.shape[1]
    q = store_theta.shape[1]

    nuhhat, Shhat = _ig_fit(store_Sigh)
    thetahat, thetastd, thetapre = _gauss_fit(store_theta)
    h0hat, h0std, h0pre = _gauss_fit(store_h0)

    big_Sigh = _ig_draws(nuhhat, Shhat, M, r)
    big_theta = _gauss_draws(thetahat, thetastd, M, r)
    big_h0 = _gauss_draws(h0hat, h0std, M, r)

    store_w = np.zeros(M)
    cIS = -.5 * (q + n) * np.log(2 * np.pi) - np.sum(np.log(np.diag(thetastd))) \
        - np.sum(np.log(np.diag(h0std))) + nuhhat @ np.log(Shhat) - np.sum(gammaln(nuhhat))

    def gIS(the, sh, b0):
        d, d0 = the - thetahat, b0 - h0hat
        return cIS - .5 * d @ thetapre @ d - (nuhhat + 1) @ np.log(sh) - np.sum(Shhat / sh) \
            - .5 * d0 @ h0pre @ d0

    for loop in range(M):
        theta = big_theta[loop]
        Sigh = big_Sigh[loop]
        h0 = big_h0[loop]

        store_w[loop] = intlike_cvarsv(Y, theta, Sigh, bigX, h0, rng=r) \
            + prior(theta, Sigh, h0) - gIS(theta, Sigh, h0)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)
