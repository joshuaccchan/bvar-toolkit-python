import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .._rng import as_rng
from .._struct import Struct
from ..sv._csv_armh import csv_armh
from ._intlike_t_csv import intlike_t_csv
from ._kron_bvar import _grid, _grid_den, _lmeanexp, _ltn_const, _niw_post, _s2
from ._kron_bvar_csv import _lden_sigh2, _rho_step
from ._kron_bvar_t import _nu_lden
from ._lniwpdf import lniwpdf


def kron_bvar_t_csv(shortY, X, pri, est, R=1000, rng=None):
    """bvt.ml.kron_bvar_t_csv - log marginal likelihood of the BVAR-t-CSV model of
    Chan (2020, JBES): integrated likelihood by importance sampling
    (bvt.ml.intlike_t_csv, lam analytic and h by IS) at the posterior means,
    prior ordinate, posterior ordinates for (A,Sig), sigh2 and nu
    Rao-Blackwellized over the stored (h, lam, rho) draws, and a rho ordinate
    from a reduced run of nsims sweeps re-drawing (h, lam, rho) at fixed
    (A_mean, Sig_mean, nu_mean, sigh2_mean), continuing from the final stored
    draws.

      ML, out = bvt.ml.kron_bvar_t_csv(shortY, X, pri, est)
      ML, out = bvt.ml.kron_bvar_t_csv(shortY, X, pri, est, R=1000, rng=None)

      pri: A0, VA0, nu0, S0, rho0, Vrho, nuh0, Sh0, nuub
      est: nsims, store_A, store_Sig (running sums), store_h, store_lam,
           store_theta ([nu rho sigh2] columns)
      R:   importance-sampling draws (default 1000)
      rng: None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out: llike, lpri, lpost, store_lpost (reduced-run den_rho column),
           store_lpost1, A_mean, Sig_mean, theta_mean

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
    rho0, Vrho, nuh0, Sh0, nuub = pri['rho0'], pri['Vrho'], pri['nuh0'], pri['Sh0'], pri['nuub']
    nsims = int(est['nsims'])
    store_h = np.asarray(est['store_h'], dtype=float).reshape(nsims, -1)
    store_lam = np.asarray(est['store_lam'], dtype=float).reshape(nsims, -1)
    store_theta = np.asarray(est['store_theta'], dtype=float).reshape(nsims, -1)

        # posterior means from the estimation run
    A_mean = np.asarray(est['store_A'], dtype=float) / nsims
    Sig_mean = np.asarray(est['store_Sig'], dtype=float) / nsims
    theta_mean = np.mean(store_theta, axis=0)

    nu_mean, rho_mean, sigh2_mean = theta_mean[0], theta_mean[1], theta_mean[2]
    llike = intlike_t_csv(shortY, X, A_mean, Sig_mean, rho_mean, sigh2_mean, nu_mean, R,
                          rng=r)[0]
    lpri = np.log(1 / (nuub - 2)) \
        - .5 * np.log(2 * np.pi * Vrho) + _ltn_const(rho0, Vrho) \
        - .5 * (rho_mean - rho0)**2 / Vrho \
        + nuh0 * np.log(Sh0) - gammaln(nuh0) - (nuh0 + 1) * np.log(sigh2_mean) - Sh0 / sigh2_mean \
        + lniwpdf(A_mean, Sig_mean, A0, sp.diags_array(1 / VA0), nu0, S0)

        # evaluate the posterior density
    store_lpost = np.zeros((nsims, 3))          # log densities of [A Sig, sigh2, nu]
    nugrid, nuidx = _grid(nu_mean, 2, nuub, 700)

    for isim in range(nsims):
        h = store_h[isim]
        lam = store_lam[isim]
        rho = store_theta[isim, 1]

            # compute the conditional density of Sig and A
        Ahat, KA, Shat = _niw_post(X, shortY, np.exp(-h) / lam, A0, VA0, S0)
        lden_ASig = lniwpdf(A_mean, Sig_mean, Ahat, KA, nu0 + T, Shat)

            # compute the conditional density of sigh2
        lden_sigh2 = _lden_sigh2(sigh2_mean, h, rho, nuh0, Sh0)

            # compute the conditional density of nu
        fnu = _nu_lden(T, np.sum(np.log(lam)), np.sum(1 / lam))
        den_nu = np.log(_grid_den(fnu(nugrid), nuidx, nugrid[1] - nugrid[0]))

        store_lpost[isim] = [lden_ASig, lden_sigh2, den_nu]
    lpost = _lmeanexp(store_lpost)

        # rho ordinate: a reduced run, continuing the chain from the last stored draws
    store_lpost1 = store_lpost
    store_lpost = np.zeros(nsims)               # [density of rho]
    rhogrid, rhoidx = _grid(rho_mean, -.999, .999, 700)
    CSig = np.linalg.cholesky(Sig_mean)
    s2u = _s2(shortY - X @ A_mean, CSig)
    nu = nu_mean
    sigh2 = sigh2_mean
    for isim in range(nsims):
        h = csv_armh(s2u / lam, rho, sigh2, h, n, rng=r)[0]

            # sample lam
        lam = 1 / r.gamrnd((n + nu) / 2, 2 / (s2u / np.exp(h) + nu))

            # sample rho; the reduced run bounds rho at .999, estimation at .9999
        rho, _, lrho = _rho_step(h, rho, sigh2, rho0, Vrho, .999, r)

            # compute the conditional density of rho
        store_lpost[isim] = _grid_den(lrho(rhogrid), rhoidx, rhogrid[1] - rhogrid[0])
    lpost = np.r_[lpost, np.log(np.mean(store_lpost))]
    ML = llike + lpri - np.sum(lpost)
    return ML, Struct(llike=llike, lpri=lpri, lpost=lpost, store_lpost=store_lpost,
                      store_lpost1=store_lpost1, A_mean=A_mean, Sig_mean=Sig_mean,
                      theta_mean=theta_mean)
