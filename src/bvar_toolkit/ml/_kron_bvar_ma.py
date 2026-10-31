import numpy as np
import scipy.sparse as sp

from .._linalg import ma_solve
from .._optim import fminbnd, fminunc
from .._struct import Struct
from ._kron_bvar import _grid, _grid_den, _lmeanexp, _ltn_const, _niw_post, _s2
from ._llike_ma import llike_ma
from ._lniwpdf import lniwpdf


def _lpri_psi(psi0, Vpsi):
    """The psi log-prior kernel, truncated to |psi| < .99."""
    def lpri_psi(x):
        return -.5 * (x - psi0)**2 / Vpsi - 1e10 * (x < -.99 or x > .99)
    return lpri_psi


def _psi_step(lp_psi, psi, psihat, Kpsic, isim, r):
    """One Metropolis-Hastings step for psi with log target lp_psi, proposing from
    N(psihat, 1/Kpsic) at the mode psihat of the target. At sweep isim = 0 and every 100th
    sweep a search from the last mode updates the mode and sets Kpsic to the second
    derivative there, or to 1/.05^2 when that is not positive; the other sweeps update the
    mode alone, by a bounded search on (-.99, .99). Returns psi, psihat and Kpsic."""
    if (isim + 1) % 100 == 0 or isim == 0:      # get the Hessian every 100 iterations
        psihat, _, hess = fminunc(lambda x: -lp_psi(x), psihat)
        Kpsic = hess if hess > 0 else 1 / .05**2
    else:
        psihat = fminbnd(lambda x: -lp_psi(x), -.99, .99)[0]
    psic = psihat + 1 / np.sqrt(Kpsic) * r.randn()
    if abs(psic) < .99:
        alpMH = lp_psi(psic) - lp_psi(psi) + \
            -.5 * (psi - psihat)**2 * Kpsic + .5 * (psic - psihat)**2 * Kpsic
    else:
        alpMH = -np.inf
    if alpMH > np.log(r.rand()):
        psi = psic
    return psi, psihat, Kpsic


def kron_bvar_ma(shortY, X, pri, est, bugcompat=False):
    """bvt.ml.kron_bvar_ma - log marginal likelihood of the BVAR-MA model by
    Chib's method. Deterministic given the stores: consumes no rng.

      ML, out = bvt.ml.kron_bvar_ma(shortY, X, pri, est)
      ML, out = bvt.ml.kron_bvar_ma(shortY, X, pri, est, bugcompat=False)

      pri: A0, VA0, nu0, S0, psi0, Vpsi
      est: nsims, store_A, store_Sig, store_psi, state.psi
           (state.psi is read only under bugcompat=True)
      out: llike, lpri, lpost, store_lpost, den_psi, A_mean, Sig_mean,
           psi_mean, psi_llike, bugcompat

    Known legacy defect, reproduced by bugcompat=True: the legacy script
    uses the estimation run's leftover final psi draw (est.state.psi) in the
    first observation's variance correction, where psi_mean is intended and
    used everywhere else. The default path uses psi_mean throughout.

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    shortY = np.asarray(shortY, dtype=float)
    X = np.asarray(X, dtype=float)
    T, n = shortY.shape
    A0, VA0 = np.asarray(pri['A0'], dtype=float), np.asarray(pri['VA0'], dtype=float).ravel()
    nu0, S0 = pri['nu0'], np.asarray(pri['S0'], dtype=float)
    psi0, Vpsi = pri['psi0'], pri['Vpsi']
    nsims = int(est['nsims'])
    store_psi = np.asarray(est['store_psi'], dtype=float).ravel()

        # estimation-tail posterior means
    A_mean = np.asarray(est['store_A'], dtype=float) / nsims
    Sig_mean = np.asarray(est['store_Sig'], dtype=float) / nsims
    psi_mean = float(np.mean(store_psi))
    lpri_psi = _lpri_psi(psi0, Vpsi)

        # evaluate the log likelihood
    CSig = np.linalg.cholesky(Sig_mean)
    s2 = _s2(ma_solve(psi_mean, shortY - X @ A_mean), CSig)
    if bugcompat:
            # Legacy defect reproduced: the estimation run's final chain draw of
            # psi stands in for psi_mean here
        psi_llike = float(est['state']['psi'])
    else:
        psi_llike = psi_mean    # corrected: consistent evaluation point
    llike = -T * n / 2 * np.log(2 * np.pi) - T * np.sum(np.log(np.diag(CSig))) \
        - n / 2 * np.log(1 + psi_mean**2) - .5 * (s2[0] / (1 + psi_llike**2) + np.sum(s2[1:]))
    lpri = lniwpdf(A_mean, Sig_mean, A0, sp.diags_array(1 / VA0), nu0, S0) \
        - .5 * np.log(2 * np.pi * Vpsi) + _ltn_const(psi0, Vpsi) - .5 * (psi_mean - psi0)**2 / Vpsi

        # evaluate the posterior density
    store_lpost = np.zeros(nsims)                # [log density of A Sig]
    for isim in range(nsims):
        psi = store_psi[isim]

            # compute the conditional density of Sig and A
        Xtld = ma_solve(psi, X)
        Ytld = ma_solve(psi, shortY)
        w = np.r_[1 / (1 + psi**2), np.ones(T - 1)]
        Ahat, KA, Shat = _niw_post(Xtld, Ytld, w, A0, VA0, S0)
        store_lpost[isim] = lniwpdf(A_mean, Sig_mean, Ahat, KA, nu0 + T, Shat)
    lpost = [_lmeanexp(store_lpost)]

        # psi ordinate: grid-normalized conditional density
    U = shortY - X @ A_mean
    psigrid, psiidx = _grid(psi_mean, -.99, .99, 700)
    tmpden = np.array([llike_ma(x, U, Sig_mean) + lpri_psi(x) for x in psigrid])
    den_psi = _grid_den(tmpden, psiidx, psigrid[1] - psigrid[0])
    lpost = np.array(lpost + [np.log(den_psi)])
    ML = llike + lpri - np.sum(lpost)
    return ML, Struct(llike=llike, lpri=lpri, lpost=lpost, store_lpost=store_lpost,
                      den_psi=den_psi, A_mean=A_mean, Sig_mean=Sig_mean, psi_mean=psi_mean,
                      psi_llike=psi_llike, bugcompat=bugcompat)
