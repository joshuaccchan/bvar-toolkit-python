import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .._linalg import ma_solve
from .._rng import as_rng
from .._struct import Struct
from ..sv._csv_armh import csv_armh
from ._intlike_csv_ma import intlike_csv_ma
from ._kron_bvar import _grid, _grid_den, _lmeanexp, _ltn_const, _niw_post, _s2
from ._kron_bvar_csv import _lden_sigh2, _rho_step
from ._kron_bvar_ma import _lpri_psi, _psi_step
from ._llike_csv_ma import llike_csv_ma
from ._lniwpdf import lniwpdf


def kron_bvar_csv_ma(shortY, X, pri, est, R=5000, nsims2=1000, rng=None):
    """bvt.ml.kron_bvar_csv_ma - log marginal likelihood of the BVAR-CSV-MA model
    of Chan (2020, JBES): integrated likelihood by importance sampling
    (bvt.ml.intlike_csv_ma) at the posterior means, prior ordinate, posterior
    ordinates for (A,Sig) and sigh2 Rao-Blackwellized over the stored
    (h, psi, rho) draws - with Hpsi REBUILT from the stored psi draw at every
    iteration - and rho / psi ordinates from a reduced run of nsims2 sweeps
    re-drawing (h, rho, psi) at fixed (A_mean, Sig_mean, sigh2_mean),
    continuing from the last stored draws with the psi-MH warm-started at the
    estimation run's final psihat.

      ML, out = bvt.ml.kron_bvar_csv_ma(shortY, X, pri, est)
      ML, out = bvt.ml.kron_bvar_csv_ma(shortY, X, pri, est, R=5000, nsims2=1000, rng=None)

      pri:    A0, VA0, nu0, S0, psi0, Vpsi, rho0, Vrho, nuh0, Sh0
      est:    nsims, store_A, store_Sig (running sums), store_h,
              store_theta ([psi rho sigh2] columns), state.psihat (the estimation
              run's final psi-MH mode, which warm starts the psi proposal)
      R:      importance-sampling draws (default 5000)
      nsims2: reduced-run length (default 1000)
      rng:    None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out:    llike, lpri, lpost, store_lpost ([den_rho den_psi] reduced-run
              columns), store_lpost1, A_mean, Sig_mean, theta_mean

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
    psi0, Vpsi = pri['psi0'], pri['Vpsi']
    rho0, Vrho, nuh0, Sh0 = pri['rho0'], pri['Vrho'], pri['nuh0'], pri['Sh0']
    nsims, nsims2 = int(est['nsims']), int(nsims2)
    store_h = np.asarray(est['store_h'], dtype=float).reshape(nsims, -1)
    store_theta = np.asarray(est['store_theta'], dtype=float).reshape(nsims, -1)
    psihat = float(est['state']['psihat'])      # estimation run's final psi-MH mode
    lpri_psi = _lpri_psi(psi0, Vpsi)            # psi log-prior kernel

        # estimation-tail posterior means
    A_mean = np.asarray(est['store_A'], dtype=float) / nsims
    Sig_mean = np.asarray(est['store_Sig'], dtype=float) / nsims
    theta_mean = np.mean(store_theta, axis=0)

    llike = intlike_csv_ma(shortY, X, A_mean, Sig_mean, theta_mean[0], theta_mean[1],
                           theta_mean[2], R, rng=r)[0]
    ngrid = 300

    lpri = -.5 * np.log(2 * np.pi * Vpsi) + _ltn_const(psi0, Vpsi) \
        - .5 * (theta_mean[0] - psi0)**2 / Vpsi \
        - .5 * np.log(2 * np.pi * Vrho) + _ltn_const(rho0, Vrho) \
        - .5 * (theta_mean[1] - rho0)**2 / Vrho \
        + nuh0 * np.log(Sh0) - gammaln(nuh0) - (nuh0 + 1) * np.log(theta_mean[2]) \
        - Sh0 / theta_mean[2] \
        + lniwpdf(A_mean, Sig_mean, A0, sp.diags_array(1 / VA0), nu0, S0)

        # evaluate the posterior density
    store_lpost = np.zeros((nsims, 2))          # [log density of A Sig, log density of sigh2]
    for isim in range(nsims):
        h = store_h[isim]
        psi = store_theta[isim, 0]
        rho = store_theta[isim, 1]

            # compute the conditional density of Sig and A
        w = np.r_[1 / (1 + psi**2) * np.exp(-h[0]), np.exp(-h[1:])]
        Ahat, KA, Shat = _niw_post(ma_solve(psi, X), ma_solve(psi, shortY), w, A0, VA0, S0)
        lden_ASig = lniwpdf(A_mean, Sig_mean, Ahat, KA, nu0 + T, Shat)

            # compute the conditional density of sigh2
        lden_sigh2 = _lden_sigh2(theta_mean[2], h, rho, nuh0, Sh0)

        store_lpost[isim] = [lden_ASig, lden_sigh2]
    lpost = _lmeanexp(store_lpost)

        # rho and psi ordinates: reduced run; h/psi/rho continue from the
        # last stored draws, psihat from the estimation run
    store_lpost1 = store_lpost
    store_lpost = np.zeros((nsims2, 2))         # [density of rho, density of psi]
    rhogrid, rhoidx = _grid(theta_mean[1], -.999, .999, ngrid)
    psigrid, psiidx = _grid(theta_mean[0], -.99, .99, ngrid)

    U = shortY - X @ A_mean
    CSig = np.linalg.cholesky(Sig_mean)
    sigh2 = theta_mean[2]
    Kpsic = None
    for isim in range(nsims2):
        s2 = _s2(ma_solve(psi, U), CSig)
        s2[0] = s2[0] / (1 + psi**2)
        h = csv_armh(s2, rho, sigh2, h, n, rng=r)[0]

            # sample rho; the reduced run bounds rho at .9999, matching estimation
        rho, _, lrho = _rho_step(h, rho, sigh2, rho0, Vrho, .9999, r)

        # sample psi
        def lp_psi(x):
            return llike_csv_ma(x, U, Sig_mean, h) + lpri_psi(x)

        psi, psihat, Kpsic = _psi_step(lp_psi, psi, psihat, Kpsic, isim, r)

            # compute the conditional density of rho
        den_rho = _grid_den(lrho(rhogrid), rhoidx, rhogrid[1] - rhogrid[0])

            # compute the conditional density of psi
        tmpden = np.array([lp_psi(x) for x in psigrid])
        den_psi = _grid_den(tmpden, psiidx, psigrid[1] - psigrid[0])

        store_lpost[isim] = [den_rho, den_psi]
    lpost = np.r_[lpost, np.log(np.mean(store_lpost, axis=0))]

    ML = llike + lpri - np.sum(lpost)
    return ML, Struct(llike=llike, lpri=lpri, lpost=lpost, store_lpost=store_lpost,
                      store_lpost1=store_lpost1, A_mean=A_mean, Sig_mean=Sig_mean,
                      theta_mean=theta_mean)
