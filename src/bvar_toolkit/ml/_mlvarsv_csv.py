import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from .._linalg import BandChol
from .._rng import as_rng
from .._stats import gamfit
from .._struct import Struct
from ..priors import niw
from ..util import ldet, mgammaln, tnormrnd
from ._isden_arss import isden_arss
from ._lgampdf import lgampdf
from ._lmvnpdf_pcn import lmvnpdf_pcn
from ._ltnormpdf import ltnormpdf


def mlvarsv_csv(X, Y, Y0, M, Hyper, store_h, store_hpara, store_kappa, is_kappafixed,
                rng=None):
    """bvt.ml.mlvarsv_csv - log marginal likelihood of the VAR-CSV model by adaptive
    importance sampling.
    (A,Sig) and the log-volatility variance sigh2 are integrated out analytically;
    h, phi and kappa are drawn from importance densities fitted to the posterior
    draws - h from bvt.ml.isden_arss, phi truncated normal, kappa gamma by
    moment/ML fit - and the M log weights are averaged in 50 batches, which also
    gives the numerical standard error.

      lml, lmlstd, out = bvt.ml.mlvarsv_csv(X, Y, Y0, M, Hyper, store_h, store_hpara,
                                            store_kappa, is_kappafixed)
      lml, lmlstd, out = bvt.ml.mlvarsv_csv(..., rng)

      Hyper: A0, VA, nu0, S0, c0, nuh, Sh, phi0, Vphi. VA is recomputed inside
             from each kappa draw before any read, so whichever version the
             caller passes is irrelevant.
      store_h     - nsim x T log-volatility draws
      store_hpara - nsim x 2, columns [phi sig2]
      store_kappa - nsim-vector
      rng         - None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out: store_w, bigml (the 50 batch values), and the fitted IS parameters

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    r = as_rng(rng)
    kappa3 = 100                    # overall shrinkage parameter for the intercept
    M = int(50 * np.ceil(M / 50))
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    Hyper = Struct(Hyper)
    k = np.shape(Hyper['A0'])[0]
    p = (k - 1) // n
    store_hpara = np.asarray(store_hpara, dtype=float)
    store_kappa = np.asarray(store_kappa, dtype=float).ravel()

    phi_hat = np.mean(store_hpara[:, 0])
    Kphi_hat = 1 / np.var(store_hpara[:, 0], ddof=1)
    big_phi = tnormrnd(phi_hat, 1 / Kphi_hat, -1, 1, M, rng=r)

    if not is_kappafixed:
        tmp_hat = gamfit(store_kappa)
        ckappa_hat = np.array([tmp_hat[0], 1 / tmp_hat[1]])
        big_kappa = r.gamrnd(ckappa_hat[0], 1 / ckappa_hat[1], M)

    h_hat, Kh_hat = isden_arss(store_h)[:2]
    CKh_hat = BandChol(Kh_hat)

    def prior(xr, xk=None):
        lp = ltnormpdf(xr, Hyper['phi0'], Hyper['Vphi'], -1, 1)
        return lp if xk is None else lp + lgampdf(xk, Hyper['c0'][0], Hyper['c0'][1])

    def gIS(xh, xr, xk=None):
        lg = lmvnpdf_pcn(xh, h_hat, Kh_hat) + ltnormpdf(xr, phi_hat, 1 / Kphi_hat, -1, 1)
        return lg if xk is None else lg + lgampdf(xk, ckappa_hat[0], ckappa_hat[1])

    store_w = np.zeros(M)
    for isim in range(M):
        phi = big_phi[isim]
        kappa = store_kappa[0] if is_kappafixed else big_kappa[isim]
            # update Hyper.VA given the new kappa
        Hyper['VA'] = niw(p, [kappa, kappa3], Y0, Y, 'mlvarsv_ncp')[1]

        h = h_hat + CKh_hat.solve_t(r.randn(T))
        lclike = (_like_var_csv(Y, X, Hyper, h) - T / 2 * np.log(2 * np.pi)
                  + .5 * np.log(1 - phi**2) + Hyper['nuh'] * np.log(Hyper['Sh'])
                  - gammaln(Hyper['nuh']) + gammaln(Hyper['nuh'] + T / 2)
                  - (Hyper['nuh'] + T / 2) * np.log(Hyper['Sh'] + .5 * _hish_quad(h, phi)))
        if is_kappafixed:
            store_w[isim] = lclike + prior(phi) - gIS(h, phi)
        else:
            store_w[isim] = lclike + prior(phi, kappa) - gIS(h, phi, kappa)
    shortw = store_w.reshape((M // 50, 50), order='F')
    maxw = shortw.max(axis=0)
    bigml = np.log(np.mean(np.exp(shortw - maxw), axis=0)) + maxw
    lml = np.mean(bigml)
    lmlstd = np.std(bigml, ddof=1) / np.sqrt(50)

    out = Struct(store_w=store_w, bigml=bigml, M=M, h_hat=h_hat, Kh_hat=Kh_hat, phi_hat=phi_hat,
                 Kphi_hat=Kphi_hat, ckappa_hat=None if is_kappafixed else ckappa_hat)
    return lml, lmlstd, out


def _hish_quad(h, phi):
    """h' Hphi' diag(1-phi^2, 1, ..., 1) Hphi h, with Hphi = I - phi*(lag)."""
    u = np.r_[h[0] * np.sqrt(1 - phi**2), h[1:] - phi * h[:-1]]
    return u @ u


def _like_var_csv(Y, X, Hyper, h):
    """log p(Y | h, kappa) with (A,Sig) integrated out under the natural-conjugate
    prior - the matric-t ordinate."""
    T, n = Y.shape
    k = np.shape(Hyper['A0'])[0]
    VA = np.asarray(Hyper['VA'], dtype=float)
    A0 = np.asarray(Hyper['A0'], dtype=float)
    XiOh = X.T * np.exp(-h)
    K_A = XiOh @ X
    K_A[np.diag_indices(k)] += 1 / VA
    CK_A = np.linalg.cholesky(K_A)
    A_hat = solve_triangular(CK_A, solve_triangular(CK_A, A0 / VA[:, None] + XiOh @ Y,
                                                    lower=True), lower=True, trans='T')
    S_hat = Hyper['S0'] + A0.T @ (A0 / VA[:, None]) + (Y.T * np.exp(-h)) @ Y \
        - A_hat.T @ K_A @ A_hat
    S_hat = (S_hat + S_hat.T) / 2                      # adjust for rounding errors
    return (-n * T / 2 * np.log(np.pi) - n / 2 * np.sum(h)
            - n / 2 * (np.sum(np.log(VA)) + 2 * np.sum(np.log(np.diag(CK_A))))
            + Hyper['nu0'] / 2 * ldet(Hyper['S0']) - (Hyper['nu0'] + T) / 2 * ldet(S_hat)
            + mgammaln(n, (Hyper['nu0'] + T) / 2) - mgammaln(n, Hyper['nu0'] / 2))
