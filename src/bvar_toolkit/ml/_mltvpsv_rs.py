import numpy as np
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ..util._dirirnd import dirirnd
from ._dirifit import dirifit
from ._intlike_rsvar import intlike_rsvar
from ._ldiripdf import ldiripdf
from ._mltvpsv_cvar import (_batch_ml, _draws, _gauss_draws, _gauss_fit, _ig_draws, _ig_fit,
                            _regressors)


def _rs_ml(Y, store_theta, store_Sig, store_P, prior, bigX, M, bugcompat, rng,
           theta_switches, Sig_switches):
    """The importance sampling estimator of the three RS-VAR models; theta_switches and
    Sig_switches say whether the coefficients and the variances have one set per regime, or
    one set that all regimes share."""
    r = as_rng(rng)
    M = int(20 * np.ceil(M / 20))
    Y = np.asarray(Y, dtype=float).ravel()
    store_theta, store_Sig = _draws(store_theta), _draws(store_Sig)
    store_P = np.asarray(store_P, dtype=float)
    nr = int(round(np.sqrt(store_P.size / store_theta.shape[0])))       # number of regimes
    store_P = store_P.reshape(-1, nr, nr)
    n = store_Sig.shape[1] // nr if Sig_switches else store_Sig.shape[1]
    k = store_theta.shape[1] // nr if theta_switches else store_theta.shape[1]
    T = Y.size // n
    bigX = _regressors(bigX, Y.size)

    shortY = Y.reshape(T, n)
    nuhat, Shat = _ig_fit(store_Sig)
    thetahat, thetastd, thetapre = _gauss_fit(store_theta)
    alphat = np.array([dirifit(store_P[:, i, :])[0] for i in range(nr)])

    big_Sig = _ig_draws(nuhat, Shat, M, r)
    big_theta = _gauss_draws(thetahat, thetastd, M, r)
    big_P = np.zeros((M, nr, nr))
    for i in range(nr):
        big_P[:, i, :] = dirirnd(alphat[i], M, rng=r)

    store_w = np.zeros(M)
    if bugcompat:
        p1 = 1 / 3
    else:
        p1 = 1 / nr

    cIS = -.5 * thetahat.size * np.log(2 * np.pi) - np.sum(np.log(np.diag(thetastd))) \
        + nuhat @ np.log(Shat) - np.sum(gammaln(nuhat))

    def gIS(the, s):
        d = the - thetahat
        return cIS - .5 * d @ thetapre @ d - (nuhat + 1) @ np.log(s) - np.sum(Shat / s)

    for isim in range(M):
        Sig = big_Sig[isim]
        theta = big_theta[isim]
        P = big_P[isim]
        g_IS_P = 0.0
        for i in range(nr):
            g_IS_P = g_IS_P + ldiripdf(P[i], alphat[i])
        thetas = theta.reshape((k, nr), order='F') if theta_switches \
            else np.tile(theta[:, None], (1, nr))
        Sigs = Sig.reshape((n, nr), order='F') if Sig_switches else np.tile(Sig[:, None], (1, nr))
        llike = intlike_rsvar(shortY, bigX, thetas, Sigs, P, p1)
        store_w[isim] = llike + prior(theta, Sig, P) - (gIS(theta, Sig) + g_IS_P)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)


def mltvpsv_rs(Y, store_theta, store_Sig, store_P, prior, bigX, M, bugcompat=False, rng=None):
    """bvt.ml.mltvpsv_rs - log marginal likelihood of the RS-VAR model of Chan and Eisenstat
    (2018) by importance sampling. The importance density of each variance is
    inverse-gamma, that of the coefficients Gaussian and that of each row of the
    transition matrix Dirichlet, fitted to the posterior draws; the regimes are integrated
    out by bvt.ml.intlike_rsvar. The M importance weights are split into 20 batches: lml
    is the mean of the 20 batch estimates of the log marginal likelihood, and lmlstd is
    their standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_rs(Y, store_theta, store_Sig, store_P, prior, bigX,
                                           M, bugcompat)
      lml, lmlstd, out = bvt.ml.mltvpsv_rs(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw; store_P is ndraws x r x r
      prior  : log prior density, prior(theta, Sig, P)
      M      : importance draws, rounded up to a multiple of 20
      rng    : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out    : store_w, the M log weights; bigml, the 20 batch estimates
      bugcompat : False (default) starts the Hamilton filter from p(s_1 = j) = 1/r;
              True reproduces the published routine, which starts from 1/3 whatever r
              is and so shifts every log likelihood, and the log marginal likelihood,
              by log(r/3).

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    return _rs_ml(Y, store_theta, store_Sig, store_P, prior, bigX, M, bugcompat, rng,
                  True, True)
