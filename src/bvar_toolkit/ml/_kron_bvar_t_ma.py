import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .._linalg import ma_solve
from .._rng import as_rng
from .._struct import Struct
from ._kron_bvar import _grid, _grid_den, _lmeanexp, _ltn_const, _niw_post, _s2
from ._kron_bvar_ma import _lpri_psi, _psi_step
from ._kron_bvar_t import _nu_lden
from ._llike_csv_ma import llike_csv_ma
from ._lniwpdf import lniwpdf


def kron_bvar_t_ma(shortY, X, pri, est, rng=None):
    """bvt.ml.kron_bvar_t_ma - log marginal likelihood of the BVAR-t-MA model of
    Chan (2020, JBES): analytic Student-t MA(1) likelihood at the posterior
    means (lam integrated analytically), prior ordinate, posterior ordinates
    for (A,Sig) and nu Rao-Blackwellized over the stored (lam, psi) draws, and
    a psi ordinate from a reduced run of nsims sweeps re-drawing (lam, psi) at
    fixed (A_mean, Sig_mean, nu_mean), continuing from the final stored psi
    draw with the psi-MH proposal warm-started at the ESTIMATION run's final
    mode psihat.

      ML, out = bvt.ml.kron_bvar_t_ma(shortY, X, pri, est)
      ML, out = bvt.ml.kron_bvar_t_ma(shortY, X, pri, est, rng=None)

      pri: A0, VA0, nu0, S0, psi0, Vpsi, nuub
      est: nsims, store_A, store_Sig (running sums), store_lam,
           store_theta ([psi nu] columns), state.psihat (the estimation run's
           final psi-MH mode, which warm starts the psi proposal)
      rng: None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out: llike, lpri, lpost, store_lpost (reduced-run den_psi column),
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
    psi0, Vpsi, nuub = pri['psi0'], pri['Vpsi'], pri['nuub']
    nsims = int(est['nsims'])
    store_lam = np.asarray(est['store_lam'], dtype=float).reshape(nsims, -1)
    store_theta = np.asarray(est['store_theta'], dtype=float).reshape(nsims, -1)
    psihat = float(est['state']['psihat'])      # estimation run's final psi-MH mode
    lpri_psi = _lpri_psi(psi0, Vpsi)            # psi log-prior kernel

        # estimation-tail posterior means
    A_mean = np.asarray(est['store_A'], dtype=float) / nsims
    Sig_mean = np.asarray(est['store_Sig'], dtype=float) / nsims
    theta_mean = np.mean(store_theta, axis=0)

    psi_mean = theta_mean[0]
    nu_mean = theta_mean[1]
    ngrid = 300

        # evaluate the log likelihood
    CSig = np.linalg.cholesky(Sig_mean)
    s2 = _s2(ma_solve(psi_mean, shortY - X @ A_mean), CSig)
    s2[0] = s2[0] / (1 + psi_mean**2)
    llike = -T * n / 2 * np.log(nu_mean * np.pi) - n / 2 * np.log(1 + psi_mean**2) \
        + T * (gammaln((nu_mean + n) / 2) - gammaln(nu_mean / 2)) \
        - T * np.sum(np.log(np.diag(CSig))) - (nu_mean + n) / 2 * np.sum(np.log(1 + s2 / nu_mean))

    lpri = lniwpdf(A_mean, Sig_mean, A0, sp.diags_array(1 / VA0), nu0, S0) \
        + np.log(1 / (nuub - 2)) \
        - .5 * np.log(2 * np.pi * Vpsi) + _ltn_const(psi0, Vpsi) - .5 * (psi_mean - psi0)**2 / Vpsi

        # evaluate the posterior density
    store_lpost = np.zeros((nsims, 2))          # [log density of A Sig, density of nu]
    nugrid, nuidx = _grid(nu_mean, 2, nuub, ngrid)
    for isim in range(nsims):
        lam = store_lam[isim]
        psi1 = store_theta[isim, 0]

            # compute the conditional density of Sig and A
        Ahat, KA, Shat = _niw_post(ma_solve(psi1, X), ma_solve(psi1, shortY), 1 / lam,
                                   A0, VA0, S0)
        lden_ASig = lniwpdf(A_mean, Sig_mean, Ahat, KA, nu0 + T, Shat)

            # compute the conditional density of nu
        fnu = _nu_lden(T, np.sum(np.log(lam)), np.sum(1 / lam))
        den_nu = _grid_den(fnu(nugrid), nuidx, nugrid[1] - nugrid[0])

        store_lpost[isim] = [lden_ASig, den_nu]
    lpost = [_lmeanexp(store_lpost[:, 0]), np.log(np.mean(store_lpost[:, 1]))]

        # psi ordinate: reduced run; psi1 continues from the last stored
        # draw, psihat from the estimation run's final mode
    store_lpost1 = store_lpost
    store_lpost = np.zeros(nsims)               # density of psi
    psigrid, psiidx = _grid(psi_mean, -.99, .99, ngrid)

    U = shortY - X @ A_mean
    Kpsic = None
    for isim in range(nsims):

        # sample lam
        s2 = _s2(ma_solve(psi1, U), CSig)
        s2[0] = s2[0] / (1 + psi1**2)
        lam = 1 / r.gamrnd((n + nu_mean) / 2, 2 / (s2 + nu_mean))

        # sample psi
        loglam = np.log(lam)

        def lp_psi(x):
            return llike_csv_ma(x, U, Sig_mean, loglam) + lpri_psi(x)

        psi1, psihat, Kpsic = _psi_step(lp_psi, psi1, psihat, Kpsic, isim, r)

            # compute the conditional density of psi [normalization centered at
            # tmpden[psiidx]; mathematically the same as max-centering]
        tmpden = np.array([lp_psi(x) for x in psigrid])
        store_lpost[isim] = _grid_den(tmpden, psiidx, psigrid[1] - psigrid[0], center=psiidx)
    lpost = np.array(lpost + [np.log(np.mean(store_lpost))])
    ML = llike + lpri - np.sum(lpost)
    return ML, Struct(llike=llike, lpri=lpri, lpost=lpost, store_lpost=store_lpost,
                      store_lpost1=store_lpost1, A_mean=A_mean, Sig_mean=Sig_mean,
                      theta_mean=theta_mean)
