import numpy as np
import scipy.sparse as sp

from .._optim import fminbnd


def isden_arss(store_h):
    """bvt.ml.isden_arss - Gaussian importance-sampling density for one log-volatility
    path, fitted to its posterior draws under the state space parameterization
      h_t = a_t + rho h_{t-1} + u_t,  u_t ~ N(0,b_t).
    rho is the maximizer of the concentrated likelihood over (-.99,.99); the
    per-period a_t, b_t are the mean and variance of the rho-differenced draws.
    Returns the mean h_hat and the PRECISION Kh_hat (band, sparse).

      h_hat, Kh_hat, r_hat, m_hat, v_hat = bvt.ml.isden_arss(store_h)

      store_h : R x T matrix of posterior draws (one row per draw)

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    store_h = np.asarray(store_h, dtype=float)
    T = store_h.shape[1]
    r_hat = fminbnd(lambda x: -_concen_like_h(store_h, x)[0], -.99, .99)[0]
    _, m_hat, v_hat = _concen_like_h(store_h, r_hat)
    h_hat = np.empty(T)                               # Hr \ m_hat, Hr = I - r_hat*(lag)
    h_hat[0] = m_hat[0]
    for t in range(1, T):
        h_hat[t] = m_hat[t] + r_hat * h_hat[t - 1]
    Hr = _hrho(T, r_hat)
    Kh_hat = (Hr.T @ sp.diags_array(1 / v_hat) @ Hr).tocsc()
    return h_hat, Kh_hat, r_hat, m_hat, v_hat


def _hrho(T, rho):
    return sp.csc_array(sp.eye_array(T) - rho * sp.eye_array(T, k=-1))


def _concen_like_h(store_h, rho):
    R, T = store_h.shape
    Hrhoh = store_h.T.copy()                          # Hrho @ store_h.T
    Hrhoh[1:] -= rho * store_h.T[:-1]
    m_hat = Hrhoh.mean(axis=1)
    E = Hrhoh - m_hat[:, None]
    v_hat = np.mean(E**2, axis=1)
    concen_like = -T * R / 2 * np.log(2 * np.pi) - R / 2 * np.sum(np.log(v_hat)) \
        - .5 * np.sum(E**2 / v_hat[:, None])
    return concen_like, m_hat, v_hat
