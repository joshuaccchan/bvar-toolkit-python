import numpy as np
from scipy.special import ndtr, ndtri

from .._rng import as_rng


def sv0_params(h, phi, Hyper, phi_bnd=.99, proposal='untruncated', rng=None):
    """bvt.sv.sv0_params - posterior draw of the SV state-equation parameters
    (phi, sig2) for stationary zero-mean AR(1) log-volatilities.

      phi, sig2, flag_phi = bvt.sv.sv0_params(h, phi, Hyper)
      phi, sig2, flag_phi = bvt.sv.sv0_params(h, phi, Hyper, phi_bnd)
      phi, sig2, flag_phi = bvt.sv.sv0_params(h, phi, Hyper, phi_bnd, proposal='truncated')

      h        : T x n zero-mean log-volatility paths, one column per series; a T-vector
                 is one series, and then phi, sig2 and flag_phi are scalars
      phi      : n-vector of current AR(1) coefficients; the new draw on output
      Hyper    : Struct or dict with nuh, Sh (inverse-gamma prior on sig2) and phi0,
                 Vphi (normal prior on phi)
      phi_bnd  : truncation bound on the phi candidate, which can be accepted
                 only if |phic| < phi_bnd; default .99, the OISV canonical value
      proposal : the Metropolis-Hastings candidate for phi. 'untruncated'
                 (default) draws it from the normal part of its conditional,
                 N(phi_hat, 1/Kphi), and rejects it unless |phic| < phi_bnd;
                 'truncated' draws it from that normal truncated to
                 (-phi_bnd, phi_bnd). Both target the same conditional.
      rng      : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      sig2     : n-vector of innovation variances
      flag_phi : n-vector, 1 where the phi candidate was accepted

    NEVER merge with bvt.sv.sv_params. This is not that sampler at mu = 0: the
    OISV pair keeps the zero-mean sampler separate, with a different truncation
    bound (.99 here, .999 there).

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    if phi_bnd is None:
        phi_bnd = .99      # OISV canonical truncation bound
    proposal = str(proposal).lower()
    if proposal not in ('untruncated', 'truncated'):
        raise ValueError(f"proposal must be 'untruncated' or 'truncated'; got '{proposal}'")
    r = as_rng(rng)
    h = np.asarray(h, dtype=float)
    one = h.ndim == 1
    if one:
        h = h[:, None]
    T, n = h.shape
    phi = np.array(phi, dtype=float).ravel() * np.ones(n)

        # sample sig2
    e_h = np.vstack([h[0] * np.sqrt(1 - phi**2), h[1:] - phi * h[:-1]])
    sig2 = 1 / r.gamrnd(Hyper['nuh'] + T / 2, 1 / (Hyper['Sh'] + np.sum(e_h**2, axis=0) / 2))
    sig2 = np.ravel(sig2) * np.ones(n)

        # sample phi: its conditional is N(phi_hat, 1/Kphi) times exp(g_phi), the part
        # of the density of h[0] that depends on phi, on |phi| < phi_bnd
    Kphi = 1 / Hyper['Vphi'] + np.sum(h[:T - 1]**2, axis=0) / sig2
    phi_hat = (Hyper['phi0'] / Hyper['Vphi'] + np.sum(h[:T - 1] * h[1:T], axis=0) / sig2) / Kphi
    Kphi = np.ravel(Kphi) * np.ones(n)
    phi_hat = np.ravel(phi_hat) * np.ones(n)
    flag_phi = np.zeros(n)

    def g_phi(x, ii):
        return .5 * np.log(1 - x**2) - .5 * (1 - x**2) / sig2[ii] * h[0, ii]**2

    if proposal == 'untruncated':
        phic = phi_hat + 1 / np.sqrt(Kphi) * r.randn(n)
        for ii in range(n):
            if abs(phic[ii]) < phi_bnd:
                alpMH = np.exp(g_phi(phic[ii], ii) - g_phi(phi[ii], ii))
                if alpMH > r.rand():
                    phi[ii] = phic[ii]
                    flag_phi[ii] = 1
    else:
        for ii in range(n):
            phic = _tnorm_draw(phi_hat[ii], 1 / np.sqrt(Kphi[ii]), -phi_bnd, phi_bnd, r)
            if np.exp(g_phi(phic, ii) - g_phi(phi[ii], ii)) > r.rand():
                phi[ii] = phic
                flag_phi[ii] = 1
    if one:
        return float(phi[0]), float(sig2[0]), float(flag_phi[0])
    return phi, sig2, flag_phi


def _tnorm_draw(mu, s, a, b, r):
    """One draw from N(mu, s^2) truncated to (a, b), by the inverse transform on the side of
    the interval nearest the mean, so an interval deep in a tail keeps its precision; beyond
    about 37 standard deviations, where the normal cdf underflows, by the exponential
    rejection sampler of Robert, C.P. (1995), Simulation of Truncated Normal Variables,
    Statistics and Computing, 5(2): 121-125."""
    alpha = (a - mu) / s
    beta = (b - mu) / s
    if beta < 0:                                       # the interval lies below the mean
        Fa = ndtr(alpha)
        Fb = ndtr(beta)
        if Fb > 0:
            z = ndtri(Fa + r.rand() * (Fb - Fa))
        else:
            z = -_tail_draw(-beta, -alpha, r)
    elif alpha > 0:                                    # the interval lies above the mean
        Ga = ndtr(-alpha)
        Gb = ndtr(-beta)
        if Ga > 0:
            z = -ndtri(Gb + r.rand() * (Ga - Gb))
        else:
            z = _tail_draw(alpha, beta, r)
    else:
        Fa = ndtr(alpha)
        Fb = ndtr(beta)
        z = ndtri(Fa + r.rand() * (Fb - Fa))
    return mu + s * z


def _tail_draw(c, d, r):
    """N(0,1) truncated to (c, d) with c far in the upper tail."""
    lam = (c + np.sqrt(c**2 + 4)) / 2
    while True:
        z = c - np.log(r.rand()) / lam
        if z < d and r.rand() <= np.exp(-(z - lam)**2 / 2):
            return z
