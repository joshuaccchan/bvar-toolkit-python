import numpy as np
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ._intlike_tvp import intlike_tvp
from ._mltvpsv_cvar import _batch_ml, _draws, _gauss_draws, _gauss_fit, _ig_draws, _ig_fit


def mltvpsv_tvp(Y, store_Sig, store_Sigtheta, store_theta0, prior, bigX, M, rng=None):
    """bvt.ml.mltvpsv_tvp - log marginal likelihood of the TVP model of Chan and Eisenstat
    (2018) by importance sampling. The importance density of each variance is
    inverse-gamma and that of the initial state Gaussian, fitted to the posterior draws;
    the states are integrated out by bvt.ml.intlike_tvp. The M importance weights are
    split into 20 batches: lml is the mean of the 20 batch estimates of the log marginal
    likelihood, and lmlstd is their standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_tvp(Y, store_Sig, store_Sigtheta, store_theta0,
                                            prior, bigX, M)
      lml, lmlstd, out = bvt.ml.mltvpsv_tvp(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw
      prior  : log prior density, prior(Sig, Sigtheta, theta0)
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
    store_Sigtheta = _draws(store_Sigtheta)
    k = store_Sigtheta.shape[1]

    nuthetahat, Sthetahat = _ig_fit(store_Sigtheta)
    nuhat, Shat = _ig_fit(store_Sig)

    theta0hat, theta0std, theta0pre = _gauss_fit(store_theta0)

    big_Sigtheta = _ig_draws(nuthetahat, Sthetahat, M, r)
    big_Sig = _ig_draws(nuhat, Shat, M, r)
    big_theta0 = _gauss_draws(theta0hat, theta0std, M, r)

    store_w = np.zeros(M)
    cIS = nuthetahat @ np.log(Sthetahat) - np.sum(gammaln(nuthetahat)) \
        + nuhat @ np.log(Shat) - np.sum(gammaln(nuhat)) - .5 * k * np.log(2 * np.pi) \
        - np.sum(np.log(np.diag(theta0std)))

    def gIS(s, sthe, a0):
        d0 = a0 - theta0hat
        return cIS - (nuthetahat + 1) @ np.log(sthe) - np.sum(Sthetahat / sthe) \
            - (nuhat + 1) @ np.log(s) - np.sum(Shat / s) - .5 * d0 @ theta0pre @ d0

    for loop in range(M):
        Sigtheta = big_Sigtheta[loop]
        Sig = big_Sig[loop]
        theta0 = big_theta0[loop]

        llike = intlike_tvp(Y, Sig, Sigtheta, bigX, theta0)
        store_w[loop] = llike + prior(Sig, Sigtheta, theta0) - gIS(Sig, Sigtheta, theta0)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)
