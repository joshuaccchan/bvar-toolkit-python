import numpy as np
from scipy.linalg import cholesky_banded
from scipy.linalg.lapack import dtbtrs

from .._rng import as_rng

# the 7-component normal mixture of Kim, Shephard and Chib (1998)
P_N = np.array([0.0073, .10556, .00002, .04395, .34001, .24566, .2575])
M_N = np.array([-10.12999, -3.97281, -8.56686, 2.77786, .61942, 1.79518, -1.08819]) - 1.2704
SIG2_N = np.array([5.79596, 2.61369, 5.17950, .16735, .64009, .34023, 1.26261])


def ksc_ar1_mean(ystar, h, mu, rho, sig2, rng=None):
    """bvt.sv.ksc_ar1_mean - KSC auxiliary-mixture sampler for the log-volatility
    path, stationary AR(1)-with-mean state equation:
      ystar_t = h_t + eps_t,  eps_t approximated by the Kim-Shephard-Chib (1998)
                7-component normal mixture,
      h_t = mu + rho*(h_{t-1} - mu) + v_t,  v_t ~ N(0,sig2),
      h_1 ~ N(mu, sig2/(1-rho^2))  (stationary initialization).

      h, S = bvt.sv.ksc_ar1_mean(ystar, h, mu, rho, sig2)
      h, S = bvt.sv.ksc_ar1_mean(ystar, h, mu, rho, sig2, rng)

      ystar is a T-vector; h is the current path, and the new draw is returned;
      sig2 is a VARIANCE. S is the T-vector of mixture indicators, in 0, ..., 6,
      drawn alongside h. rng is None, an integer seed, a numpy.random.Generator or a
      bvt.Stream.

    NOT interchangeable with the random-walk variants ksc_rw_h0 / ksc_rw_diffuse:
    different state equations.

    See:
    Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood
    Inference and Comparison with ARCH Models, Review of Economic Studies, 65(3):
    361-393.
    Chan, J.C.C. and Jeliazkov, I. (2009). Efficient Simulation and Integrated
    Likelihood Estimation in State Space Models, International Journal of
    Mathematical Modelling and Numerical Optimisation, 1(1/2): 101-120.
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for
    Large Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    r = as_rng(rng)
    ystar = np.asarray(ystar, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    T = h.size

        # sample S from a 7-point discrete distribution
    tmprand = r.rand(T)
    sd = np.sqrt(SIG2_N)
    q = P_N * (np.exp(-0.5 * ((ystar[:, None] - (h[:, None] + M_N)) / sd)**2)
               / (np.sqrt(2 * np.pi) * sd))
    q = q / q.sum(axis=1, keepdims=True)
    S = 7 - np.sum(tmprand[:, None] < np.cumsum(q, axis=1), axis=1)

        # sample h using the precision sampler
    # y^* = h + d + epsilon, epsilon ~ N(0, Omega)
    # h  ~ N(mu 1_T, sig2 (Hrho' S^{-1} Hrho)^{-1}), where S^{-1} = diag(1-rho^2, 1, ..., 1);
    # HiSH = Hrho' S^{-1} Hrho is tridiagonal with diagonal HiSH_d and off-diagonal -rho
    dv = np.r_[1 - rho**2, np.ones(T - 1)]
    HiSH_d = dv.copy()
    HiSH_d[:-1] = dv[:-1] + (-rho * dv[1:]) * (-rho)
    HiSH_e = -rho * dv[1:]
    d = M_N[S]
    iOmega = 1 / SIG2_N[S]
    ab = np.zeros((2, T))                              # lower band storage of Kh
    ab[0] = HiSH_d / sig2 + iOmega
    ab[1, :-1] = HiSH_e / sig2
    CKh = cholesky_banded(ab if T > 1 else ab[:1], lower=True)
    HiSH_1 = HiSH_d.copy()                             # HiSH @ ones(T)
    HiSH_1[:-1] += HiSH_e
    HiSH_1[1:] += HiSH_e
    rhs = mu / sig2 * HiSH_1 + iOmega * (ystar - d)
    v = dtbtrs(CKh, rhs[:, None], uplo='L')[0]
    h_hat = dtbtrs(CKh, v, uplo='L', trans='T')[0][:, 0]
    h = h_hat + dtbtrs(CKh, r.randn(T)[:, None], uplo='L', trans='T')[0][:, 0]
    return h, S
