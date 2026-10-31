import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .._linalg import ma_solve
from .._rng import as_rng
from .._struct import Struct
from ..sv._csv_armh import csv_armh
from ._intlike_csv_t_ma import intlike_csv_t_ma
from ._kron_bvar import _grid, _grid_den, _lmeanexp, _ltn_const, _niw_post, _s2
from ._kron_bvar_csv import _lden_sigh2, _rho_step
from ._kron_bvar_ma import _lpri_psi, _psi_step
from ._kron_bvar_t import _nu_lden
from ._llike_csv_ma import llike_csv_ma
from ._lniwpdf import lniwpdf


def kron_bvar_csv_t_ma(shortY, X, pri, est, bugcompat=False, R=10000, nsims2=1000, rng=None):
    """bvt.ml.kron_bvar_csv_t_ma - log marginal likelihood of the BVAR-CSV-t-MA
    model by Chib's method: integrated likelihood by importance sampling,
    Rao-Blackwellized ordinates for (A,Sig), sigh2 and nu, and rho / psi
    ordinates from a reduced run at the posterior means.

      ML, out = bvt.ml.kron_bvar_csv_t_ma(shortY, X, pri, est)
      ML, out = bvt.ml.kron_bvar_csv_t_ma(shortY, X, pri, est, bugcompat=False, R=10000,
                                          nsims2=1000, rng=None)

      pri:       A0, VA0, nu0, S0, psi0, Vpsi, rho0, Vrho, nuh0, Sh0, nuub
      est:       nsims, store_A, store_Sig (running sums), store_h, store_lam,
                 store_theta ([psi rho sigh2 nu] columns), state.psihat (the estimation
                 run's final psi-MH mode, which warm starts the psi proposal), state.psi
                 and state.Sig (final chain draws; required under bugcompat)
      bugcompat: reproduce the legacy defects below (default False)
      R:         importance-sampling draws (default 10000)
      nsims2:    reduced-run length (default 1000)
      rng:       None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out:       llike, lpri, lpost, store_lpost ([den_rho den_psi] reduced-run
                 columns), store_lpost1, A_mean, Sig_mean, theta_mean, bugcompat

    Two known legacy defects, both reproduced by bugcompat=True:
    (1) the (A,Sig) ordinate loop never refreshes psi or rebuilds Hpsi, so all
        nsims terms condition on the leftover final draw;
    (2) the reduced run's psi target passes the leftover final Sig where
        Sig_mean is intended.
    The default path fixes both. Quirks kept in either mode (rho bound .999,
    psigrid +/-.999, ngrid 299, the lam step's missing (1+psi^2) correction, the
    reduced run's warm start) are deliberate sampler details.

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
    rho0, Vrho, nuh0, Sh0, nuub = pri['rho0'], pri['Vrho'], pri['nuh0'], pri['Sh0'], pri['nuub']
    nsims, nsims2 = int(est['nsims']), int(nsims2)
    store_h = np.asarray(est['store_h'], dtype=float).reshape(nsims, -1)
    store_lam = np.asarray(est['store_lam'], dtype=float).reshape(nsims, -1)
    store_theta = np.asarray(est['store_theta'], dtype=float).reshape(nsims, -1)
    psihat = float(est['state']['psihat'])      # estimation run's final psi-MH mode
    lpri_psi = _lpri_psi(psi0, Vpsi)            # psi log-prior kernel

    ngrid = 299

    A_mean = np.asarray(est['store_A'], dtype=float) / nsims
    Sig_mean = np.asarray(est['store_Sig'], dtype=float) / nsims
    theta_mean = np.mean(store_theta, axis=0)
    psi_mean = theta_mean[0]
    rho_mean = theta_mean[1]
    sigh2_mean = theta_mean[2]
    nu_mean = theta_mean[3]

    llike = intlike_csv_t_ma(shortY, X, A_mean, Sig_mean, psi_mean, rho_mean, sigh2_mean,
                             nu_mean, R, rng=r)[0]
    lpri = -.5 * np.log(2 * np.pi * Vpsi) + _ltn_const(psi0, Vpsi) \
        - .5 * (theta_mean[0] - psi0)**2 / Vpsi \
        + np.log(1 / (nuub - 2)) \
        - .5 * np.log(2 * np.pi * Vrho) + _ltn_const(rho0, Vrho) \
        - .5 * (rho_mean - rho0)**2 / Vrho \
        + nuh0 * np.log(Sh0) - gammaln(nuh0) - (nuh0 + 1) * np.log(sigh2_mean) - Sh0 / sigh2_mean \
        + lniwpdf(A_mean, Sig_mean, A0, sp.diags_array(1 / VA0), nu0, S0)

        # evaluate the posterior density
    store_lpost = np.zeros((nsims, 3))          # log densities of [A Sig, sigh2, nu]
    nugrid, nuidx = _grid(nu_mean, 2, nuub, 700)

    if bugcompat:
            # Legacy defect (1) reproduced: psi and Hpsi frozen at the
            # estimation run's final chain values for the whole ordinate loop
        psi = float(est['state']['psi'])
    for isim in range(nsims):
        h = store_h[isim]
        lam = store_lam[isim]
        rho = store_theta[isim, 1]
        if not bugcompat:
                # corrected: integrate over the stored psi draws, the pattern
                # bvt.ml.kron_bvar_csv_ma follows
            psi = store_theta[isim, 0]

            # compute the conditional density of Sig and A
        w = np.r_[1 / (1 + psi**2) * np.exp(-h[0]), np.exp(-h[1:])] / lam
        Ahat, KA, Shat = _niw_post(ma_solve(psi, X), ma_solve(psi, shortY), w, A0, VA0, S0)
        lden_ASig = lniwpdf(A_mean, Sig_mean, Ahat, KA, nu0 + T, Shat)

            # compute the conditional density of sigh2
        lden_sigh2 = _lden_sigh2(sigh2_mean, h, rho, nuh0, Sh0)

            # compute the conditional density of nu
        fnu = _nu_lden(T, np.sum(np.log(lam)), np.sum(1 / lam))
        den_nu = np.log(_grid_den(fnu(nugrid), nuidx, nugrid[1] - nugrid[0]))

        store_lpost[isim] = [lden_ASig, lden_sigh2, den_nu]
    lpost = _lmeanexp(store_lpost)

        # rho and psi ordinates: reduced run; (h, lam, rho) continue from the
        # last stored draws, psi from the final chain value
        # (= store_theta[nsims-1, 0] in both modes), psihat from the estimation run
    store_lpost1 = store_lpost
    store_lpost = np.zeros((nsims2, 2))         # [density of rho, density of psi]
    rhogrid, rhoidx = _grid(rho_mean, -.999, .999, ngrid)
    psigrid, psiidx = _grid(theta_mean[0], -.999, .999, ngrid)
    U = shortY - X @ A_mean
    CSig = np.linalg.cholesky(Sig_mean)
    nu = nu_mean
    sigh2 = sigh2_mean
    if bugcompat:
            # Legacy defect (2) reproduced: the psi target conditions on the
            # final estimation DRAW of Sig, not Sig_mean
        Sig_psi = np.asarray(est['state']['Sig'], dtype=float)
    else:
        Sig_psi = Sig_mean
    if not bugcompat:
            # entry state after the corrected ordinate loop: psi carries the
            # last stored draw - the same value the bugcompat path froze
        psi = store_theta[nsims - 1, 0]

    Kpsic = None
    for isim in range(nsims2):
        s2u = _s2(ma_solve(psi, U), CSig)
        s2_h = s2u / lam
        s2_h[0] = s2_h[0] / (1 + psi**2)
        h = csv_armh(s2_h, rho, sigh2, h, n, rng=r)[0]

            # sample lam [no (1+psi^2) first-observation correction, mirroring
            # the estimation sampler]
        lam = 1 / r.gamrnd((n + nu) / 2, 2 / (s2u / np.exp(h) + nu))

            # sample rho [reduced-run bound .999; the estimation run uses .99]
        rho, _, lrho = _rho_step(h, rho, sigh2, rho0, Vrho, .999, r)

            # sample psi
        U_psi = U / np.sqrt(lam)[:, None]

        def lp_psi(x):
            return llike_csv_ma(x, U_psi, Sig_psi, h) + lpri_psi(x)

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
                      theta_mean=theta_mean, bugcompat=bugcompat)
