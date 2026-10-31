import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from .._rng import as_rng
from .._stats import gamfit
from .._struct import Struct


def _draws(store):
    """The posterior draws as a matrix, one row per draw."""
    store = np.asarray(store, dtype=float)
    return store.reshape(store.shape[0], -1)


def _regressors(X, N):
    """The regressors as a matrix of N rows: a sparse matrix as it is, and an array reshaped, so
    that a single regressor may come as a vector."""
    return X if sp.issparse(X) else np.asarray(X, dtype=float).reshape(N, -1)


def _ig_fit(store):
    """The inverse-gamma importance density (nu, S) of each column of variance draws, from the
    gamma fit to their inverses."""
    store = _draws(store)
    tmp = np.array([gamfit(1 / store[:, i]) for i in range(store.shape[1])])
    return tmp[:, 0], 1 / tmp[:, 1]


def _gauss_fit(store):
    """The Gaussian importance density of the draws: their mean, the lower Cholesky factor of
    their covariance and the inverse of that covariance."""
    store = _draws(store)
    hat = np.mean(store, axis=0)
    std = np.linalg.cholesky(np.atleast_2d(np.cov(store, rowvar=False)))
    pre = solve_triangular(std.T, solve_triangular(std, np.eye(hat.size), lower=True))
    return hat, std, pre


def _ig_draws(nu, S, M, r):
    """M draws from each inverse-gamma importance density, one column each."""
    return np.column_stack([1 / r.gamrnd(nu[i], 1 / S[i], M) for i in range(nu.size)])


def _gauss_draws(hat, std, M, r):
    """M draws from the Gaussian importance density, one row each."""
    return hat + (std @ r.randn(hat.size, M)).T


def _batch_ml(store_w):
    """The mean and the standard error of the 20 batch estimates of the log marginal likelihood
    from the log importance weights, and the batch estimates."""
    M = store_w.size
    shortw = store_w.reshape((M // 20, 20), order='F')
    maxw = shortw.max(axis=0)
    bigml = np.log(np.mean(np.exp(shortw - maxw), axis=0)) + maxw
    return np.mean(bigml), np.std(bigml, ddof=1) / np.sqrt(20), bigml


def mltvpsv_cvar(Y, store_theta, store_Sig, prior, bigX, M, rng=None):
    """bvt.ml.mltvpsv_cvar - log marginal likelihood of the CVAR model of Chan and Eisenstat
    (2018) by importance sampling. The importance density of each variance is
    inverse-gamma and that of the coefficients Gaussian, fitted to the posterior draws;
    the likelihood is Gaussian. The M importance weights are split into 20 batches: lml is
    the mean of the 20 batch estimates of the log marginal likelihood, and lmlstd is their
    standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_cvar(Y, store_theta, store_Sig, prior, bigX, M)
      lml, lmlstd, out = bvt.ml.mltvpsv_cvar(..., rng)

      Y      : T*n data, stacked by period
      store_*: posterior draws, one row per draw
      prior  : log prior density, prior(theta, Sig)
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
    store_theta, store_Sig = _draws(store_theta), _draws(store_Sig)
    n = store_Sig.shape[1]
    k = store_theta.shape[1]
    T = Y.size // n
    bigX = _regressors(bigX, Y.size)

    nuhat, Shat = _ig_fit(store_Sig)
    thetahat, thetastd, thetapre = _gauss_fit(store_theta)

    big_Sig = _ig_draws(nuhat, Shat, M, r)
    big_theta = _gauss_draws(thetahat, thetastd, M, r)

    store_w = np.zeros(M)

    cIS = -.5 * k * np.log(2 * np.pi) - np.sum(np.log(np.diag(thetastd))) \
        + nuhat @ np.log(Shat) - np.sum(gammaln(nuhat))

    def gIS(the, s):
        d = the - thetahat
        return cIS - .5 * d @ thetapre @ d - (nuhat + 1) @ np.log(s) - np.sum(Shat / s)

    def like_svar(the, s):
        e = Y - bigX @ the
        return -T * n / 2 * np.log(2 * np.pi) - T / 2 * np.sum(np.log(s)) \
            - .5 * e @ (e / np.tile(s, T))

    for loop in range(M):
        Sig = big_Sig[loop]
        theta = big_theta[loop]
        llike = like_svar(theta, Sig)
        store_w[loop] = llike + prior(theta, Sig) - gIS(theta, Sig)
    lml, lmlstd, bigml = _batch_ml(store_w)
    return lml, lmlstd, Struct(store_w=store_w, bigml=bigml)
