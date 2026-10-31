import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ._intlike_tvpsv import intlike_tvpsv
from ._mltvpsv_cvar import (_batch_ml, _draws, _gauss_draws, _gauss_fit, _ig_draws, _ig_fit,
                            _regressors)


def mltvpsv_tvp_r3_sv(Y, store_beta, store_gam, store_Sigmu, store_Sigh, store_h0, store_mu0,
                      prior, Z, M, rng=None):
    """bvt.ml.mltvpsv_tvp_r3_sv - log marginal likelihood of the TVP-R3-SV model of Chan and
    Eisenstat (2018) by importance sampling. The importance density of each variance is
    inverse-gamma and that of (beta, gam) and of each initial state Gaussian, fitted to
    the posterior draws; the intercepts mu_t and the log-volatilities are integrated out
    by bvt.ml.intlike_tvpsv, applied to Y - Z @ np.r_[beta, gam] with the identity as the
    regressors. The M importance weights are split into 20 batches: lml is the mean of the
    20 batch estimates of the log marginal likelihood, and lmlstd is their standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_tvp_r3_sv(Y, store_beta, store_gam, store_Sigmu,
                                                  store_Sigh, store_h0, store_mu0, prior, Z, M)
      lml, lmlstd, out = bvt.ml.mltvpsv_tvp_r3_sv(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw
      prior  : log prior density, prior(beta, gam, Sigmu, Sigh, mu0, h0)
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
    store_beta, store_gam = _draws(store_beta), _draws(store_gam)
    store_Sigmu, store_Sigh = _draws(store_Sigmu), _draws(store_Sigh)
    n = store_Sigh.shape[1]
    T = Y.size // n
    kbeta = store_beta.shape[1]
    kgam = store_gam.shape[1]
    Z = _regressors(Z, Y.size)

    numuhat, Smuhat = _ig_fit(store_Sigmu)
    nuhhat, Shhat = _ig_fit(store_Sigh)
    betahat, betastd, betapre = _gauss_fit(np.hstack([store_beta, store_gam]))

    mu0hat, mu0std, mu0pre = _gauss_fit(store_mu0)
    h0hat, h0std, h0pre = _gauss_fit(store_h0)

    big_Sigmu = _ig_draws(numuhat, Smuhat, M, r)
    big_Sigh = _ig_draws(nuhhat, Shhat, M, r)
    big_beta = _gauss_draws(betahat, betastd, M, r)
    big_mu0 = _gauss_draws(mu0hat, mu0std, M, r)
    big_h0 = _gauss_draws(h0hat, h0std, M, r)

    store_w = np.zeros(M)
    cIS = -.5 * (kbeta + kgam) * np.log(2 * np.pi) - np.sum(np.log(np.diag(betastd))) \
        + numuhat @ np.log(Smuhat) - np.sum(gammaln(numuhat)) \
        + nuhhat @ np.log(Shhat) - np.sum(gammaln(nuhhat)) \
        - .5 * (2 * n) * np.log(2 * np.pi) - np.sum(np.log(np.diag(mu0std))) \
        - np.sum(np.log(np.diag(h0std)))

    def gIS(b, sm, sh, a0, b0):
        d, da, db = b - betahat, a0 - mu0hat, b0 - h0hat
        return cIS - .5 * d @ betapre @ d \
            - (numuhat + 1) @ np.log(sm) - np.sum(Smuhat / sm) \
            - (nuhhat + 1) @ np.log(sh) - np.sum(Shhat / sh) \
            - .5 * da @ mu0pre @ da - .5 * db @ h0pre @ db

    In = sp.eye_array(T * n, format='csc')
    for loop in range(M):
        beta = big_beta[loop, :kbeta]
        gam = big_beta[loop, kbeta:]
        Sigmu = big_Sigmu[loop]
        Sigh = big_Sigh[loop]
        mu0 = big_mu0[loop]
        h0 = big_h0[loop]
        llike = intlike_tvpsv(Y - Z @ np.r_[beta, gam], Sigmu, Sigh, In, h0, mu0, rng=r)[0]
        store_w[loop] = llike + prior(beta, gam, Sigmu, Sigh, mu0, h0) \
            - gIS(np.r_[beta, gam], Sigmu, Sigh, mu0, h0)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)
