import numpy as np
from scipy.linalg import cholesky_banded
from scipy.linalg.lapack import dtbtrs

from .._rng import as_rng
from ._ksc_ar1_mean import M_N, P_N, SIG2_N


def _indicators(ystar, h, r):
    """The mixture indicators, in 0, ..., 6, from one uniform per period."""
    tmprand = r.rand(h.size)
    sd = np.sqrt(SIG2_N)
    q = P_N * (np.exp(-0.5 * ((ystar[:, None] - (h[:, None] + M_N)) / sd)**2)
               / (np.sqrt(2 * np.pi) * sd))
    q = q / q.sum(axis=1, keepdims=True)
    return 7 - np.sum(tmprand[:, None] < np.cumsum(q, axis=1), axis=1)


def _draw(dv, iOmega, rhs, r):
    """h ~ N(K^{-1} rhs, K^{-1}) for K = H' diag(dv) H + diag(iOmega), H the first difference
    matrix, by the lower banded Cholesky factor of K."""
    T = dv.size
    ab = np.zeros((2, T))
    ab[0] = dv + iOmega
    ab[0, :-1] += dv[1:]
    ab[1, :-1] = -dv[1:]
    C = cholesky_banded(ab if T > 1 else ab[:1], lower=True)
    v = dtbtrs(C, rhs[:, None], uplo='L')[0]
    hhat = dtbtrs(C, v, uplo='L', trans='T')[0][:, 0]
    return hhat + dtbtrs(C, r.randn(T)[:, None], uplo='L', trans='T')[0][:, 0]


def ksc_rw_diffuse(ystar, h, omega2h, Vh, rng=None):
    """bvt.sv.ksc_rw_diffuse - KSC auxiliary-mixture sampler for the log-volatility
    path, random-walk state equation with a DIFFUSE-style initial condition
    h_1 ~ N(0,Vh) (Vh a variance, supplied by the caller):
      ystar_t = h_t + eps_t,  eps_t approximated by the Kim-Shephard-Chib (1998)
                7-component normal mixture,
      h_t = h_{t-1} + v_t,    v_t ~ N(0,omega2h),   h_1 ~ N(0,Vh).

      h, S = bvt.sv.ksc_rw_diffuse(ystar, h, omega2h, Vh)
      h, S = bvt.sv.ksc_rw_diffuse(ystar, h, omega2h, Vh, rng)

      ystar is a T-vector; h on input is the current path and on output the new draw;
      omega2h and Vh are VARIANCES. S is the T-vector of mixture indicators, in
      0, ..., 6, drawn alongside h. rng is None, an integer seed, a numpy.random.Generator
      or a bvt.Stream.

    Distinct from ksc_rw_h0: different initial condition, hence different draws.

    See:
    Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
    Inference and Comparison with ARCH Models, Review of Economic Studies, 65(3):
    361-393.
    Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
    Likelihood Estimation in State Space Models, International Journal of
    Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
    Chan, J.C.C. (2013). Moving Average Stochastic Volatility Models with
    Application to Inflation Forecast, Journal of Econometrics, 176(2): 162-172.
    """
    r = as_rng(rng)
    ystar = np.asarray(ystar, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    S = _indicators(ystar, h, r)
    dv = np.r_[1 / Vh, 1 / omega2h * np.ones(h.size - 1)]
    iOmega = 1 / SIG2_N[S]
    return _draw(dv, iOmega, iOmega * (ystar - M_N[S]), r), S
