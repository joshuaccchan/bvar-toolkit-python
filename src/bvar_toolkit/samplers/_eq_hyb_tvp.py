import numpy as np
import scipy.sparse as sp

from .._linalg import BandChol
from .._rng import as_rng
from ..util import surform


def _config(Xs, Sigs, iehi, Ei):
    """The regressors Z of the standardized state deviations of one configuration, the
    precision K = H'H + Z' diag(iehi) Z of their path, its factor and the mode."""
    T, m = Xs.shape
    Z = surform(Xs * np.sqrt(Sigs))
    ZSig = Z.T @ sp.diags_array(iehi)
    H = sp.eye_array(T * m, format='csc') - sp.eye_array(T * m, k=-m, format='csc')
    K = (H.T @ H + ZSig @ Z).tocsc()
    C = BandChol(K)
    xhat = C.solve_t(C.solve(ZSig @ Ei))
    return Z, K, C, xhat


def eq_hyb_tvp(Xi, thetai0, Sigthetai, Yi, hi, p0i, k, is_gamfixed, gami, rng=None):
    """bvt.samplers.eq_hyb_tvp - the hybrid TVP-VAR block for ONE equation: draws the
    pair of binary indicators gam_i = (gam^beta_i, gam^alpha_i) marginally of the
    states, then draws the state path theta_i given them. The two indicators set
    time variation on (1) or off (0) separately for the equation's VAR
    coefficients and its impact-matrix elements.

      gami, thetai, Ui, tilde_thetai, lp_gami = bvt.samplers.eq_hyb_tvp(Xi, thetai0,
          Sigthetai, Yi, hi, p0i, k, is_gamfixed, gami)
      ... = bvt.samplers.eq_hyb_tvp(..., rng)

      Xi        : T x ki regressors, [X2, -Y[:, :ii]] - lags then the
                  contemporaneous variables of the preceding equations
      thetai0   : ki-vector, the constant part of the state (the equation's intercept-level
                  coefficients); Sigthetai the ki variances of the state innovations
      hi        : T-vector log-volatility path;  p0i : the 2 prior inclusion
                  probabilities;  k : number of beta (VAR-coefficient) columns, so
                  m = ki - k is the number of alpha columns
      is_gamfixed : True holds gami at the passed value and skips the comparison
      gami      : the 2 indicators, returned as a 2-vector
      lp_gami   : the 4 log posterior ordinates of the four configurations, in the
                  order (0,0), (0,1), (1,0), (1,1)
      Ui        : T-vector equation residual net of the state path, the input to the
                  volatility block
      rng       : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    When m = 0 (the first equation) the two configurations with time-varying alpha are
    assigned a likelihood far below the others and are never drawn.

    rng consumption, in order: one rand for the gam draw (only when is_gamfixed is
    False), then randn(T*ki) for gam = (1,1), randn(T*k) for (1,0),
    randn(T*m) for (0,1), and nothing at all for (0,0).

    See:
    Chan, J.C.C. (2023). Large Hybrid Time-Varying Parameter VARs, Journal of
    Business and Economic Statistics, 41(3): 890-905.
    """
    r = as_rng(rng)
    Xi = np.asarray(Xi, dtype=float)
    thetai0 = np.asarray(thetai0, dtype=float).ravel()
    Sigthetai = np.asarray(Sigthetai, dtype=float).ravel()
    Yi = np.asarray(Yi, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    p0i = np.asarray(p0i, dtype=float).ravel()
    gami = np.array(gami, dtype=float).ravel()
    T, ki = Xi.shape
    iehi = np.exp(-hi)
    Ei = Yi - Xi @ thetai0
    m = ki - k
    lp_gami = np.zeros(4)
    blocks = {}
    if not is_gamfixed:          # if gam^theta_i is not fixed, sample gam_thetai marginal of thetai
        pri_gam = np.array([1 - p0i[0], 1 - p0i[0], p0i[0], p0i[0]]) \
            * np.array([1 - p0i[1], p0i[1], 1 - p0i[1], p0i[1]])
        c1 = -T / 2 * np.log(2 * np.pi) - .5 * np.sum(hi)
        c2 = -.5 * iehi @ (Ei**2)
            # likelihood value for gam = (0,0)
        llike00 = c1 + c2

            # compute likelihood value for gam = (0,1)
        if m > 0:
            Z, K, C, xhat = blocks['01'] = _config(Xi[:, k:], Sigthetai[k:], iehi, Ei)
            llike01 = c1 - C.logdet() / 2 + c2 + .5 * xhat @ (K @ xhat)
        else:
            llike01 = min(-1e10, llike00 - 100)

            # compute likelihood value for gam = (1,0)
        Z, K, C, xhat = blocks['10'] = _config(Xi[:, :k], Sigthetai[:k], iehi, Ei)
        llike10 = c1 - C.logdet() / 2 + c2 + .5 * xhat @ (K @ xhat)

            # compute likelihood value for gam = (1,1)
        if m > 0:
            Z, K, C, xhat = blocks['11'] = _config(Xi, Sigthetai, iehi, Ei)
            llike11 = c1 - C.logdet() / 2 + c2 + .5 * xhat @ (K @ xhat)
        else:
            llike11 = min(-1e10, llike10 - 100)

        llike = np.array([llike00, llike01, llike10, llike11])
        maxllike = np.max(llike)
        prob_gami_ker = np.exp(llike - maxllike) * pri_gam
        prob_gami = prob_gami_ker / np.sum(prob_gami_ker)
        lp_gami = llike - maxllike + np.log(pri_gam) - np.log(np.sum(prob_gami_ker))
        tmpdraw = int(np.flatnonzero(np.cumsum(prob_gami) > r.rand())[0])
        gami = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])[tmpdraw]
    else:                        # if gam^theta_i is fixed, compute a few things for sampling thetai
        if gami[0] == 1 and gami[1] == 1:
            blocks['11'] = _config(Xi, Sigthetai, iehi, Ei)
        elif gami[0] == 1 and gami[1] == 0:
            blocks['10'] = _config(Xi[:, :k], Sigthetai[:k], iehi, Ei)
        elif gami[0] == 0 and gami[1] == 1:
            blocks['01'] = _config(Xi[:, k:], Sigthetai[k:], iehi, Ei)

        # sample thetai
    mu_thetai = np.tile(thetai0, T)
    sd = np.tile(np.sqrt(Sigthetai), T)
    if gami[0] == 1 and gami[1] == 1:
        Z, K, C, xhat = blocks['11']
        tilde_thetai = xhat + C.solve_t(r.randn(T * ki))
        thetai = mu_thetai + sd * tilde_thetai
        Ui = Ei - Z @ tilde_thetai
    elif gami[0] == 1 and gami[1] == 0:
        Z, K, C, xhat = blocks['10']
        tilde_betai = xhat + C.solve_t(r.randn(T * k))
        tilde_thetai = np.hstack([tilde_betai.reshape(T, k), np.zeros((T, m))]).ravel()
        thetai = mu_thetai + sd * tilde_thetai
        Ui = Ei - Z @ tilde_betai
    elif gami[0] == 0 and gami[1] == 1:
        Z, K, C, xhat = blocks['01']
        tilde_alpi = xhat + C.solve_t(r.randn(T * m))
        tilde_thetai = np.hstack([np.zeros((T, k)), tilde_alpi.reshape(T, m)]).ravel()
        thetai = mu_thetai + sd * tilde_thetai
        Ui = Ei - Z @ tilde_alpi
    else:
        thetai = mu_thetai
        Ui = Ei
        tilde_thetai = np.zeros(T * ki)          # tilde_thetai isn't used so it's set to 0
    return gami, thetai, Ui, tilde_thetai, lp_gami
