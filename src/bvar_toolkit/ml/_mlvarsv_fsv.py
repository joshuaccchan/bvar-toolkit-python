import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.special import gammaln, ndtr

from .._linalg import BandChol
from .._rng import as_rng
from .._stats import gamfit
from .._struct import Struct
from ..priors import minn
from ..util import surform2, tnormrnd
from ._isden_arss import isden_arss
from ._mlvarsv_arsv_redu import _hish_quad


def mlvarsv_fsv(X, Y, Y0, M, Hyper, flag_marg, store_h, store_hpara, store_l, store_kappa,
                is_kappafixed, is_kappasym, gram='full', rng=None):
    """bvt.ml.mlvarsv_fsv - log marginal likelihood of the VAR-FSV model (factor SV)
    by adaptive importance sampling, with the VAR coefficients and the r latent
    factors integrated out analytically and the remaining parameters drawn from
    importance densities fitted to the posterior draws.

      lml, lmlstd, out = bvt.ml.mlvarsv_fsv(X, Y, Y0, M, Hyper, flag_marg, store_h,
          store_hpara, store_l, store_kappa, is_kappafixed, is_kappasym, gram='full',
          rng=None)

      M           - number of importance draws, rounded up here to a multiple of
                    50; lml is the mean of the 50 batch log marginal likelihoods
                    and lmlstd their standard error
      flag_marg   - 1 or 2; any other value raises an error. 2 also integrates out
                    the log-volatility variances; VAR-FSV is the only model in this
                    family that implements 1
      Hyper: alp0, Valp, c0, nuh, Sh, mu0, Vmu, phi0, Vphi, l0, Vl. Valp is
             recomputed inside from each kappa draw before any read, so whichever
             version the caller passes is irrelevant.
      store_h     - nsim x T x (n+r); r is inferred as store_h.shape[2] - n
      store_hpara - nsim x 3(n+r), columns [mu' phi' sig2']
      store_l     - nsim x kl free loadings
      store_kappa - nsim x 2
      gram        - how the weighted Gram matrix in the precision of the VAR
                    coefficients is formed: 'full' (default, as the published
                    code) or 'blocks'. The two agree to rounding and consume the
                    same random draws; 'blocks' is faster when n is large
      rng         - None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out: store_w, bigml (the 50 batch values), and the fitted IS parameters

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    if gram not in ('full', 'blocks'):
        raise ValueError("gram must be 'full' or 'blocks'")
    if flag_marg not in (1, 2):
        raise ValueError('flag_marg must be 1 or 2 (as in the legacy switch)')
    blocks = gram == 'blocks'
    r_ = as_rng(rng)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Hyper = Struct(Hyper)
    M = int(50 * np.ceil(M / 50))
    T, n = Y.shape
    store_h = np.asarray(store_h, dtype=float)
    npr = store_h.shape[2]
    r = npr - n
    k = X.shape[1]
    p = (k - 1) // n
    store_hpara = np.asarray(store_hpara, dtype=float)
    store_l = np.asarray(store_l, dtype=float).reshape(store_h.shape[0], -1)
    store_kappa = np.asarray(store_kappa, dtype=float).reshape(store_h.shape[0], -1)
    kl = store_l.shape[1]
    kappa3 = 100

        # obtain parameters for importance sampling densities
    if is_kappafixed:
        pass
    elif is_kappasym:
        tmp_hat = gamfit(store_kappa[:, 0])
        big_kappa = r_.gamrnd(tmp_hat[0], tmp_hat[1], M)[:, None]
        ckappa_hat = np.array([[tmp_hat[0], 1 / tmp_hat[1]]])
    else:
        ckappa_hat = np.zeros((2, 2))
        big_kappa = np.zeros((M, 2))
        for i in range(2):
            tmp_hat = gamfit(store_kappa[:, i])
            big_kappa[:, i] = r_.gamrnd(tmp_hat[0], tmp_hat[1], M)
            ckappa_hat[i] = [tmp_hat[0], 1 / tmp_hat[1]]
    lhat = store_l.mean(axis=0)
    lstd = np.linalg.cholesky(np.atleast_2d(np.cov(store_l, rowvar=False)))
    lpre = solve_triangular(lstd, solve_triangular(lstd, np.eye(kl), lower=True), lower=True,
                            trans='T')
    muhat = store_hpara[:, :npr].mean(axis=0)
    muvar = store_hpara[:, :npr].var(axis=0, ddof=1)
    phihat = store_hpara[:, npr:2 * npr].mean(axis=0)
    phivar = store_hpara[:, npr:2 * npr].var(axis=0, ddof=1)
    tmp = np.array([gamfit(1 / store_hpara[:, 2 * npr + i]) for i in range(npr)])
    nusig2hat, Ssig2hat = tmp[:, 0], 1 / tmp[:, 1]
    h_hat = np.zeros(T * npr)
    Kh_blocks = []
    for ii in range(npr):
        hi_hat, Khi_hat = isden_arss(store_h[:, :, ii])[:2]
        h_hat[ii * T:(ii + 1) * T] = hi_hat
        Kh_blocks.append(Khi_hat)
    Kh_hat = sp.block_diag(Kh_blocks, format='csc')         # one series at a time
    CKh_hat = BandChol(Kh_hat)

        # obtain importance sampling draws
    big_l = lhat + (lstd @ r_.randn(kl, M)).T
    big_mu = muhat + np.sqrt(muvar) * r_.randn(M, npr)
    big_phi = np.zeros((M, npr))
    for ii in range(npr):
        big_phi[:, ii] = tnormrnd(phihat[ii], phivar[ii], -1, 1, M, rng=r_)
    big_sig2 = np.zeros((M, npr))          # drawn even under flag_marg = 2, where nothing reads it
    for i in range(npr):
        big_sig2[:, i] = 1 / r_.gamrnd(nusig2hat[i], 1 / Ssig2hat[i], M)

    phi0, Vphi = np.asarray(Hyper['phi0'], dtype=float), np.asarray(Hyper['Vphi'], dtype=float)
    mu0, Vmu = np.asarray(Hyper['mu0'], dtype=float), np.asarray(Hyper['Vmu'], dtype=float)
    nuh, Sh = np.asarray(Hyper['nuh'], dtype=float), np.asarray(Hyper['Sh'], dtype=float)
    l0, Vl = float(Hyper['l0']), float(Hyper['Vl'])
    ctrunc_prior = np.sum(np.log(ndtr((1 - phi0) / np.sqrt(Vphi))
                                 - ndtr((-1 - phi0) / np.sqrt(Vphi))))
    ctrunc_IS = np.sum(np.log(ndtr((1 - phihat) / np.sqrt(phivar))
                              - ndtr((-1 - phihat) / np.sqrt(phivar))))
    base_prior = -.5 * (kl + 2 * npr) * np.log(2 * np.pi) - ctrunc_prior - kl / 2 * np.log(Vl) \
        - .5 * np.sum(np.log(Vmu)) - .5 * np.sum(np.log(Vphi))
    base_IS = -.5 * (kl + 2 * npr) * np.log(2 * np.pi) - ctrunc_IS - np.sum(np.log(np.diag(lstd))) \
        - .5 * np.sum(np.log(muvar)) - .5 * np.sum(np.log(phivar))
    if flag_marg == 1:
        cprior = base_prior + nuh @ np.log(Sh) - np.sum(gammaln(nuh))
        cIS = base_IS + nusig2hat @ np.log(Ssig2hat) - np.sum(gammaln(nusig2hat))
        ck_prior = ck_IS = None
    else:
        c0 = np.asarray(Hyper['c0'], dtype=float).reshape(-1, 2)
        if is_kappafixed:
            ck_prior = ck_IS = None
        else:
            ck_prior = c0[:1] if is_kappasym else c0[:2]
            ck_IS = ckappa_hat
        cprior = base_prior + (0 if ck_prior is None else ck_prior[:, 0] @ np.log(ck_prior[:, 1])
                               - np.sum(gammaln(ck_prior[:, 0])))
        cIS = base_IS + (0 if ck_IS is None else ck_IS[:, 0] @ np.log(ck_IS[:, 1])
                         - np.sum(gammaln(ck_IS[:, 0])))

    def prior(lv, m, ph, s=None, xk=None):
        lp = cprior - .5 * (lv - l0) @ (lv - l0) / Vl - .5 * np.sum((m - mu0)**2 / Vmu) \
            - .5 * np.sum((ph - phi0)**2 / Vphi)
        if s is not None:
            lp += -(nuh + 1) @ np.log(s) - np.sum(Sh / s)
        if xk is not None:
            lp += (ck_prior[:, 0] - 1) @ np.log(xk) - ck_prior[:, 1] @ xk
        return lp

    def gIS(lv, m, ph, s=None, xk=None):
        lg = cIS - .5 * (lv - lhat) @ lpre @ (lv - lhat) - .5 * np.sum((m - muhat)**2 / muvar) \
            - .5 * np.sum((ph - phihat)**2 / phivar)
        if s is not None:
            lg += -(nusig2hat + 1) @ np.log(s) - np.sum(Ssig2hat / s)
        if xk is not None:
            lg += (ck_IS[:, 0] - 1) @ np.log(xk) - ck_IS[:, 1] @ xk
        return lg

    store_w = np.zeros(M)
    L = np.vstack([np.eye(r), np.ones((n - r, r))])
    L_idx = np.nonzero(np.tril(np.ones((n, r)), -1).T)[::-1]   # free elements, column by column
    c_hi = np.sum(np.log(CKh_hat.ab[0]))
    for isim in range(M):
        longh = h_hat + CKh_hat.solve_t(r_.randn(T * npr))
        h = longh.reshape((T, npr), order='F')
        lv = big_l[isim]
        L[L_idx] = lv
        mu = big_mu[isim]
        phi = big_phi[isim]
        sig2 = big_sig2[isim]
        if is_kappafixed:
            kappa = store_kappa[0]
        elif is_kappasym:
            kappa = np.r_[big_kappa[isim, 0], big_kappa[isim, 0]]
        else:
            kappa = big_kappa[isim]
        alp0, Valp = minn(p, kappa[0], kappa[1], kappa3, Y0, Y, 4)[:2]
        Hyper['alp0'], Hyper['Valp'] = alp0, Valp

        lh_prior = 0.0
        for ii in range(npr):
            q = _hish_quad(h[:, ii] - mu[ii], phi[ii])
            if flag_marg == 1:
                lh_prior += -T / 2 * np.log(sig2[ii]) + .5 * np.log(1 - phi[ii]**2) \
                    - .5 / sig2[ii] * q
            else:
                lh_prior += .5 * np.log(1 - phi[ii]**2) + nuh[ii] * np.log(Sh[ii]) \
                    - gammaln(nuh[ii]) + gammaln(nuh[ii] + T / 2) \
                    - (nuh[ii] + T / 2) * np.log(Sh[ii] + .5 * q)
        dh = longh - h_hat
        lh_g = c_hi - .5 * dh @ (Kh_hat @ dh)
        llike = _deny_fsv_blocks(X, Y, L, h, Hyper) if blocks else _deny_fsv(X, Y, L, h, Hyper)
        if flag_marg == 1:
            store_w[isim] = llike + lh_prior - lh_g + prior(lv, mu, phi, s=sig2) \
                - gIS(lv, mu, phi, s=sig2)
        elif is_kappafixed:
            store_w[isim] = llike + lh_prior - lh_g + prior(lv, mu, phi) - gIS(lv, mu, phi)
        else:
            xk = kappa[:1] if is_kappasym else kappa
            store_w[isim] = llike + lh_prior - lh_g + prior(lv, mu, phi, xk=xk) \
                - gIS(lv, mu, phi, xk=xk)
    shortw = store_w.reshape((M // 50, 50), order='F')
    maxw = shortw.max(axis=0)
    bigml = np.log(np.mean(np.exp(shortw - maxw), axis=0)) + maxw
    lml = np.mean(bigml)
    lmlstd = np.std(bigml, ddof=1) / np.sqrt(50)

    out = Struct(store_w=store_w, bigml=bigml, M=M, r=r, kl=kl, h_hat=h_hat, Kh_hat=Kh_hat,
                 lhat=lhat, lstd=lstd, muhat=muhat, muvar=muvar, phihat=phihat, phivar=phivar,
                 nusig2hat=nusig2hat, Ssig2hat=Ssig2hat,
                 ckappa_hat=None if is_kappafixed else ckappa_hat)
    return lml, lmlstd, out


def _deny_fsv(X, Y, L, h, Hyper):
    """log p(Y | L, h, kappa) with the VAR coefficients and the latent factors
    integrated out."""
    T, n = Y.shape
    r = h.shape[1] - n
    bigX = surform2(X, n).toarray()
    y = Y.ravel()
    k_alp = bigX.shape[1]
    Valp, alp0 = np.asarray(Hyper['Valp'], dtype=float), np.asarray(Hyper['alp0'], dtype=float)
    IL = sp.kron(sp.eye_array(T), sp.csc_array(L), format='csc')
    Sy = IL @ sp.diags_array(np.exp(h[:, n:n + r]).ravel()) @ IL.T \
        + sp.diags_array(np.exp(h[:, :n]).ravel())
    CSy = BandChol(Sy)
    XiSy = CSy.solve_t(CSy.solve(bigX)).T
    Kalp = XiSy @ bigX
    Kalp[np.diag_indices(k_alp)] += 1 / Valp
    CKalp = np.linalg.cholesky(Kalp)
    tmpc = solve_triangular(CKalp, alp0 / Valp + XiSy @ y, lower=True)
    return (-T * n / 2 * np.log(2 * np.pi) - np.sum(np.log(CSy.ab[0]))
            - .5 * np.sum(np.log(Valp)) - np.sum(np.log(np.diag(CKalp)))
            - .5 * (y @ CSy.solve_t(CSy.solve(y)) + np.sum(alp0**2 / Valp) - np.sum(tmpc**2)))


def _deny_fsv_blocks(X, Y, L, h, Hyper):
    """_deny_fsv with bigX' inv(Sy) bigX = the sum over t of kron(P_t, x_t' x_t), where P_t
    is the inverse of the tth n x n block of Sy; Sy and bigX are not formed."""
    T, n = Y.shape
    k = X.shape[1]
    r = h.shape[1] - n
    Valp, alp0 = np.asarray(Hyper['Valp'], dtype=float), np.asarray(Hyper['alp0'], dtype=float)
    P = np.zeros((T, n * n))
    PY = np.zeros((T, n))
    yPy = 0.0
    ldSy = 0.0
    for t in range(T):
        CSt = np.linalg.cholesky((L * np.exp(h[t, n:n + r])) @ L.T + np.diag(np.exp(h[t, :n])))
        Pt = solve_triangular(CSt, solve_triangular(CSt, np.eye(n), lower=True), lower=True,
                              trans='T')
        P[t] = Pt.ravel(order='F')
        PY[t] = Y[t] @ Pt
        yPy += PY[t] @ Y[t]
        ldSy += np.sum(np.log(np.diag(CSt)))
    XX = (X[:, :, None] * X[:, None, :]).reshape(T, k * k, order='F')   # row t is vec(x_t' x_t)
    XPX = np.transpose((XX.T @ P).reshape((k, k, n, n), order='F'),
                       (0, 2, 1, 3)).reshape((n * k, n * k), order='F')
    Kalp = XPX.copy()
    Kalp[np.diag_indices(n * k)] += 1 / Valp
    CKalp = np.linalg.cholesky(Kalp)
    tmpc = solve_triangular(CKalp, alp0 / Valp + (X.T @ PY).ravel(order='F'), lower=True)
    return (-T * n / 2 * np.log(2 * np.pi) - ldSy - .5 * np.sum(np.log(Valp))
            - np.sum(np.log(np.diag(CKalp)))
            - .5 * (yPy + np.sum(alp0**2 / Valp) - np.sum(tmpc**2)))
