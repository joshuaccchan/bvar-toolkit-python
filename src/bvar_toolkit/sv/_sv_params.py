import numpy as np

from .._rng import as_rng


def sv_params(h, mu, phi, Hyper, phi_bnd=.999, rng=None):
    """bvt.sv.sv_params - posterior draw of the SV state equation parameters
    (mu, phi, sig2) for stationary AR(1) log-volatilities with mean mu.

      mu, phi, sig2, flag_phi = bvt.sv.sv_params(h, mu, phi, Hyper)
      mu, phi, sig2, flag_phi = bvt.sv.sv_params(h, mu, phi, Hyper, phi_bnd)

      h        : T x (n+r) log-volatility paths; the first n columns have mean mu
                 (n = len(mu)), the last r are zero-mean and share the same phi
                 and sig2 draws
      mu       : n-vector of current means; the new draw on output
      phi      : (n+r)-vector of current AR(1) coefficients; the new draw on output
      Hyper    : Struct or dict with nuh, Sh (inverse-gamma prior on sig2), phi0, Vphi
                 (normal prior on phi) and mu0, Vmu (normal prior on mu)
      phi_bnd  : the phi candidate is accepted only if |phic| < phi_bnd; default
                 .999, the OISV canonical value, while ml_varsv uses .998
      rng      : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      sig2     : (n+r)-vector of innovation variances
      flag_phi : (n+r)-vector, 1 where the phi candidate was accepted

    Trap: the mu block runs only when EVERY element of mu is nonzero; a single exact
    zero silently skips the mu draw for all n series.

    NEVER merge with bvt.sv.sv0_params, the separate zero-mean sampler, which
    carries a different truncation bound (.99).

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    if phi_bnd is None:
        phi_bnd = .999     # OISV canonical truncation bound
    r = as_rng(rng)
    h = np.asarray(h, dtype=float)
    if h.ndim == 1:
        h = h[:, None]
    mu = np.array(mu, dtype=float).ravel()
    n = mu.size
    T, npr = h.shape
    phi = np.array(phi, dtype=float).ravel() * np.ones(npr)

        # sample sig2
    tmp_h = np.column_stack([h[:, :n] - mu, h[:, n:]])
    e_h = np.vstack([tmp_h[0] * np.sqrt(1 - phi**2), tmp_h[1:] - phi * tmp_h[:-1]])
    sig2 = 1 / r.gamrnd(Hyper['nuh'] + T / 2, 1 / (Hyper['Sh'] + np.sum(e_h**2, axis=0) / 2))
    sig2 = np.ravel(sig2) * np.ones(npr)

        # sample phi
    Kphi = 1 / Hyper['Vphi'] + np.sum(tmp_h[:T - 1]**2, axis=0) / sig2
    phi_hat = (Hyper['phi0'] / Hyper['Vphi'] + np.sum(tmp_h[:T - 1] * tmp_h[1:T], axis=0) / sig2) \
        / Kphi
    phic = phi_hat + 1 / np.sqrt(Kphi) * r.randn(npr)
    flag_phi = np.zeros(npr)
    for ii in range(npr):
        if abs(phic[ii]) < phi_bnd:
            alpMH = np.exp(_g_phi(phic[ii], sig2[ii], tmp_h[0, ii])
                           - _g_phi(phi[ii], sig2[ii], tmp_h[0, ii]))
            if alpMH > r.rand():
                phi[ii] = phic[ii]
                flag_phi[ii] = 1

        # sample mu
    if n > 0 and np.all(mu != 0):
        p1, s1 = phi[:n], sig2[:n]
        Kmu = 1 / Hyper['Vmu'] + ((1 - p1**2) + (T - 1) * (1 - p1)**2) / s1
        mu_hat = (Hyper['mu0'] / Hyper['Vmu'] + (1 - p1**2) / s1 * h[0, :n]
                  + (1 - p1) / s1 * np.sum(h[1:, :n] - p1 * h[:-1, :n], axis=0)) / Kmu
        mu = mu_hat + 1 / np.sqrt(Kmu) * r.randn(n)
    return mu, phi, sig2, flag_phi


def _g_phi(x, sig2, h1):
    return .5 * np.log(1 - x**2) - .5 * (1 - x**2) / sig2 * h1**2
