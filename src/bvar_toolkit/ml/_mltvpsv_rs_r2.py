from ._mltvpsv_rs import _rs_ml


def mltvpsv_rs_r2(Y, store_theta, store_Sig, store_P, prior, bigX, M, bugcompat=False,
                  rng=None):
    """bvt.ml.mltvpsv_rs_r2 - log marginal likelihood of the RS-VAR-R2 model of Chan and
    Eisenstat (2018) by importance sampling. The importance density of each variance is
    inverse-gamma, that of the coefficients Gaussian and that of each row of the
    transition matrix Dirichlet, fitted to the posterior draws; the regimes are integrated
    out by bvt.ml.intlike_rsvar. The M importance weights are split into 20 batches: lml
    is the mean of the 20 batch estimates of the log marginal likelihood, and lmlstd is
    their standard error.

      lml, lmlstd, out = bvt.ml.mltvpsv_rs_r2(Y, store_theta, store_Sig, store_P, prior,
                                              bigX, M, bugcompat)
      lml, lmlstd, out = bvt.ml.mltvpsv_rs_r2(..., rng)

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
                  True, False)
