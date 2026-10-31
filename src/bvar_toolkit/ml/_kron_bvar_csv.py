import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ..sv._csv_armh import csv_armh
from ._intlike_csv import intlike_csv
from ._kron_bvar import _grid, _grid_den, _lmeanexp, _ltn_const, _niw_post, _s2
from ._linvgammpdf import linvgammpdf
from ._lniwpdf import lniwpdf


def _lden_sigh2(sigh2, h, rho, nuh0, Sh0):
    """The log conditional density at sigh2 of the innovation variance of the AR(1) path h."""
    eh = np.r_[h[0] * np.sqrt(1 - rho**2), h[1:] - rho * h[:-1]]
    return linvgammpdf(sigh2, nuh0 + h.size / 2, Sh0 + np.sum(eh**2) / 2)


def _rho_step(h, rho, sigh2, rho0, Vrho, bound, r):
    """One Metropolis-Hastings step for the AR(1) coefficient rho of h, proposing from
    N(rhohat, 1/Krho) and rejecting candidates with |rho| >= bound. Returns rho, whether it
    moved, and the log conditional density kernel of rho for its ordinate."""
    T = h.size
    Krho = 1 / Vrho + np.sum(h[:T - 1]**2) / sigh2
    rhohat = (rho0 / Vrho + h[:T - 1] @ h[1:] / sigh2) / Krho
    rhoc = rhohat + r.randn() / np.sqrt(Krho)

    def grho(x):
        return -.5 * np.log(sigh2 / (1 - x**2)) - .5 * (1 - x**2) / sigh2 * h[0]**2

    moved = False
    if abs(rhoc) < bound:
        alpMH = np.exp(grho(rhoc) - grho(rho))
        if alpMH > r.rand():
            rho, moved = rhoc, True
    return rho, moved, lambda x: grho(x) - .5 * Krho * (x - rhohat)**2


def kron_bvar_csv(shortY, X, pri, est, R=1000, rng=None):
    """bvt.ml.kron_bvar_csv - log marginal likelihood of the BVAR-CSV model of
    Chan (2020, JBES): integrated likelihood by importance sampling
    (bvt.ml.intlike_csv, h integrated out) at the posterior means, prior
    ordinate (NIW x truncated-normal rho x inverse-gamma sigh2), posterior
    ordinates for (A,Sig) and sigh2 Rao-Blackwellized over the stored (h, rho)
    draws, and a rho ordinate from a REDUCED MCMC RUN of nsims sweeps that
    re-draws (h, rho) at fixed (A_mean, Sig_mean, sigh2_mean), continuing the
    chain from the final stored draws.

      ML, out = bvt.ml.kron_bvar_csv(shortY, X, pri, est)
      ML, out = bvt.ml.kron_bvar_csv(shortY, X, pri, est, R=1000, rng=None)

      pri: A0, VA0, nu0, S0, rho0, Vrho, nuh0, Sh0
      est: nsims, store_A, store_Sig (running sums), store_h, store_theta
           ([rho sigh2] columns), state.countrho (the estimation run's rho-MH
           acceptance count, carried forward into out.countrho)
      R:   importance-sampling draws for the integrated likelihood (default 1000)
      rng: None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out: llike, lpri, lpost, store_lpost (reduced-run den_rho column),
           store_lpost1, countrho, A_mean, Sig_mean, theta_mean, h_mean

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    r = as_rng(rng)
    shortY = np.asarray(shortY, dtype=float)
    X = np.asarray(X, dtype=float)
    T, n = shortY.shape
    A0, VA0 = np.asarray(pri['A0'], dtype=float), np.asarray(pri['VA0'], dtype=float).ravel()
    nu0, S0 = pri['nu0'], np.asarray(pri['S0'], dtype=float)
    rho0, Vrho, nuh0, Sh0 = pri['rho0'], pri['Vrho'], pri['nuh0'], pri['Sh0']
    nsims = int(est['nsims'])
    store_h = np.asarray(est['store_h'], dtype=float).reshape(nsims, -1)
    store_theta = np.asarray(est['store_theta'], dtype=float).reshape(nsims, -1)
    countrho = int(est['state']['countrho'])    # continues the estimation run's counter

        # estimation-tail posterior means
    A_mean = np.asarray(est['store_A'], dtype=float) / nsims
    Sig_mean = np.asarray(est['store_Sig'], dtype=float) / nsims
    h_mean = np.mean(store_h, axis=0)
    theta_mean = np.mean(store_theta, axis=0)

    llike = intlike_csv(shortY, X, A_mean, Sig_mean, theta_mean[0], theta_mean[1], R, rng=r)[0]
    lpri = -.5 * np.log(2 * np.pi * Vrho) + _ltn_const(rho0, Vrho) \
        - .5 * (theta_mean[0] - rho0)**2 / Vrho \
        + nuh0 * np.log(Sh0) - gammaln(nuh0) - (nuh0 + 1) * np.log(theta_mean[1]) \
        - Sh0 / theta_mean[1] \
        + lniwpdf(A_mean, Sig_mean, A0, sp.diags_array(1 / VA0), nu0, S0)

        # evaluate the posterior density
    store_lpost = np.zeros((nsims, 2))          # [log density of A Sig, log density of sigh2]
    for isim in range(nsims):
        h = store_h[isim]
        rho = store_theta[isim, 0]

            # compute the conditional density of Sig and A
        Ahat, KA, Shat = _niw_post(X, shortY, np.exp(-h), A0, VA0, S0)
        lden_ASig = lniwpdf(A_mean, Sig_mean, Ahat, KA, nu0 + T, Shat)

            # compute the conditional density of sigh2
        lden_sigh2 = _lden_sigh2(theta_mean[1], h, rho, nuh0, Sh0)

        store_lpost[isim] = [lden_ASig, lden_sigh2]
    lpost = _lmeanexp(store_lpost)

        # rho ordinate: reduced run, continuing the chain from the last stored draws
    store_lpost1 = store_lpost                  # first-phase densities, for out
    store_lpost = np.zeros(nsims)               # [density of rho]
    rhogrid, rhoidx = _grid(theta_mean[0], -.999, .999, 700)
    CSig = np.linalg.cholesky(Sig_mean)
    s2 = _s2(shortY - X @ A_mean, CSig)
    sigh2 = theta_mean[1]
    for isim in range(nsims):
            # AR-MH h step: NR start at h_mean, with a forced accept on the
            # reduced run's first sweep
        h = csv_armh(s2, rho, sigh2, h, n, isim == 0, h_mean, rng=r)[0]

            # sample rho
        rho, moved, lrho = _rho_step(h, rho, sigh2, rho0, Vrho, .9999, r)
        countrho += moved

            # compute the conditional density of rho
        store_lpost[isim] = _grid_den(lrho(rhogrid), rhoidx, rhogrid[1] - rhogrid[0])
    lpost = np.r_[lpost, np.log(np.mean(store_lpost))]
    ML = llike + lpri - np.sum(lpost)
    return ML, Struct(llike=llike, lpri=lpri, lpost=lpost, store_lpost=store_lpost,
                      store_lpost1=store_lpost1, countrho=countrho, A_mean=A_mean,
                      Sig_mean=Sig_mean, theta_mean=theta_mean, h_mean=h_mean)
