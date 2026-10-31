import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.special import gammaln, ndtr

from .._linalg import BandChol
from .._rng import as_rng
from .._stats import betafit, betapdf, gamfit
from .._struct import Struct
from ..priors import impact_B0, minn
from ..util import kron_gram, tnormrnd
from ._isden_arss import isden_arss
from ._mlvarsv_arsv_redu import _hish_quad


def mlvarsv_arsvo_redu(X, Y, Y0, M, Hyper, flag_marg, store_h, store_beta, store_hpara,
                       store_kappa, store_o, store_po, o_grid, is_kappafixed, is_kappasym,
                       bugcompat=False, gram='full', rng=None):
    """bvt.ml.mlvarsv_arsvo_redu - log marginal likelihood of the VAR-SVO model
    (Cholesky SV with an outlier component) by adaptive importance sampling. As
    bvt.ml.mlvarsv_arsv_redu, with the outlier scales o_t drawn from the smoothed
    empirical posterior o_hat and the outlier probability po from a fitted beta.

      lml, lmlstd, out = bvt.ml.mlvarsv_arsvo_redu(X, Y, Y0, M, Hyper, flag_marg,
          store_h, store_beta, store_hpara, store_kappa, store_o, store_po, o_grid,
          is_kappafixed, is_kappasym, bugcompat=False, gram='full', rng=None)

      flag_marg - 2 only; any other value raises an error
      Hyper: as bvt.ml.mlvarsv_arsv_redu plus p0a, p0b (the beta prior on po)
      store_o  - nsim x T outlier scales
      store_po - nsim-vector
      o_grid   - the 32-atom grid [1, linspace(2,20,31)] the sampler used. o_hat
                 is built by exact == against store_o, so this must be the same
                 grid the chain drew from
      bugcompat - default False. True reproduces three defects of the published
                 code, all in the o block: (1) the outlier prior mass is spread
                 over all 32 grid atoms where the sampler spreads it over 31;
                 (2) the T x 32 o_hat is indexed linearly, so for T >= 32 every
                 index lands in the first column instead of at (t, o_idx[t]); (3) the
                 Jacobian -n*sum(log(o)) of the outlier scaling is omitted. The
                 default fixes all three. None of the fixes consumes rng, so both
                 modes draw the identical stream and differ only in the weights.
      gram     - as in bvt.ml.mlvarsv_arsv_redu, with the weights
                 exp(-h[:, i])/o**2
      rng      - None, an integer seed, a numpy.random.Generator or a bvt.Stream
      out: store_w, bigml, store_lr_o, store_lJ_o (the o Jacobian actually
           applied; all zeros under bugcompat), o_hat, and the fitted IS parameters

    Under is_kappasym the kappa prior is scored with rows 1 and 2 of Hyper.c0 while
    the sampler draws kappa1 from row 0, so rows 0 and 1 of c0 must be equal for
    the weights to be right.

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    if gram not in ('full', 'blocks'):
        raise ValueError("gram must be 'full' or 'blocks'")
    if flag_marg != 2:
        raise ValueError('only flag_marg = 2 is implemented (as in the legacy switch)')
    blocks = gram == 'blocks'
    r = as_rng(rng)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Hyper = Struct(Hyper)
    T, n = Y.shape
    k = X.shape[1]
    p = (k - 1) // n
    k_beta = n * (n - 1) // 2
    o_grid = np.asarray(o_grid, dtype=float).ravel()
    ngrid = o_grid.size               # number of atoms (32); the sampler ngrid is one fewer
    M = int(50 * np.ceil(M / 50))
    kappa3 = 100
    B0_id = np.tril_indices(n, -1)
    B0 = np.eye(n)
    store_h = np.asarray(store_h, dtype=float)
    store_beta = np.asarray(store_beta, dtype=float).reshape(store_h.shape[0], -1)
    store_hpara = np.asarray(store_hpara, dtype=float)
    store_kappa = np.asarray(store_kappa, dtype=float)
    store_o = np.asarray(store_o, dtype=float)
    store_po = np.asarray(store_po, dtype=float).ravel()

        # how the prior mass po is split, and how o_hat is indexed (defects 1-2)
    n_pri = ngrid if bugcompat else ngrid - 1

        # obtain parameters for importance sampling densities
    if is_kappafixed:              # kappa1 and kappa2 are fixed, but kappa4 is free
        tmp_hat = gamfit(store_kappa[:, 2])
        big_kappa4 = r.gamrnd(tmp_hat[0], tmp_hat[1], M)
        ckappa_hat = np.array([tmp_hat[0], 1 / tmp_hat[1]])
    else:                          # under is_kappasym kappa1 = kappa2: kappa2 and kappa4
        cols = [1, 2] if is_kappasym else [0, 1, 2]
        ckappa_hat = np.zeros((len(cols), 2))
        big_kappa = np.zeros((M, len(cols)))
        for j, i in enumerate(cols):
            tmp_hat = gamfit(store_kappa[:, i])
            big_kappa[:, j] = r.gamrnd(tmp_hat[0], tmp_hat[1], M)
            ckappa_hat[j] = [tmp_hat[0], 1 / tmp_hat[1]]
    betahat = store_beta.mean(axis=0)
    betacov = np.atleast_2d(np.cov(store_beta, rowvar=False))
    betaCcov = np.linalg.cholesky(betacov)
    muhat = store_hpara[:, :n].mean(axis=0)
    muvar = store_hpara[:, :n].var(axis=0, ddof=1)
    phihat = store_hpara[:, n:2 * n].mean(axis=0)
    phivar = store_hpara[:, n:2 * n].var(axis=0, ddof=1)
    h_hat = np.zeros(T * n)
    Kh_blocks = []
    for ii in range(n):
        hi_hat, Khi_hat = isden_arss(store_h[:, :, ii])[:2]
        h_hat[ii * T:(ii + 1) * T] = hi_hat
        Kh_blocks.append(Khi_hat)
    Kh_hat = sp.block_diag(Kh_blocks, format='csc')         # one series at a time
    CKh_hat = BandChol(Kh_hat)
    tmp_count = np.stack([np.sum(store_o == g, axis=0) for g in o_grid], axis=1)
    o_hat = (tmp_count + 1) / (store_o.shape[0] + ngrid)
    cumsumo_hat = np.cumsum(o_hat, axis=1)
    po_hat = betafit(store_po)

        # obtain importance sampling draws
    big_mu = muhat + np.sqrt(muvar) * r.randn(M, n)
    big_phi = np.zeros((M, n))
    for ii in range(n):
        big_phi[:, ii] = tnormrnd(phihat[ii], phivar[ii], -1, 1, M, rng=r)
    big_po = r.betarnd(po_hat[0], po_hat[1], M)

    c0 = np.asarray(Hyper['c0'], dtype=float).reshape(3, 2)
    phi0, Vphi = np.asarray(Hyper['phi0'], dtype=float), np.asarray(Hyper['Vphi'], dtype=float)
    mu0, Vmu = np.asarray(Hyper['mu0'], dtype=float), np.asarray(Hyper['Vmu'], dtype=float)
    ctrunc_prior = np.sum(np.log(ndtr((1 - phi0) / np.sqrt(Vphi))
                                 - ndtr((-1 - phi0) / np.sqrt(Vphi))))
    ctrunc_IS = np.sum(np.log(ndtr((1 - phihat) / np.sqrt(phivar))
                              - ndtr((-1 - phihat) / np.sqrt(phivar))))
    if is_kappafixed:
        ck_prior, ck_IS = c0[2:3], ckappa_hat[None, :]
    elif is_kappasym:
        ck_prior, ck_IS = c0[1:3], ckappa_hat
    else:
        ck_prior, ck_IS = c0, ckappa_hat
    cprior = -.5 * (2 * n) * np.log(2 * np.pi) - ctrunc_prior - .5 * np.sum(np.log(Vmu)) \
        - .5 * np.sum(np.log(Vphi)) + ck_prior[:, 0] @ np.log(ck_prior[:, 1]) \
        - np.sum(gammaln(ck_prior[:, 0]))
    cIS = -.5 * (2 * n) * np.log(2 * np.pi) - ctrunc_IS - .5 * np.sum(np.log(muvar)) \
        - .5 * np.sum(np.log(phivar)) + ck_IS[:, 0] @ np.log(ck_IS[:, 1]) \
        - np.sum(gammaln(ck_IS[:, 0]))

    def prior(m, ph, xk, xpo):
        return cprior - .5 * np.sum((m - mu0)**2 / Vmu) - .5 * np.sum((ph - phi0)**2 / Vphi) \
            + (ck_prior[:, 0] - 1) @ np.log(xk) - ck_prior[:, 1] @ xk \
            + np.log(betapdf(xpo, Hyper['p0a'], Hyper['p0b']))

    def gIS(m, ph, xk, xpo):
        return cIS - .5 * np.sum((m - muhat)**2 / muvar) - .5 * np.sum((ph - phihat)**2 / phivar) \
            + (ck_IS[:, 0] - 1) @ np.log(xk) - ck_IS[:, 1] @ xk \
            + np.log(betapdf(xpo, po_hat[0], po_hat[1]))

    store_w = np.zeros(M)
    store_lr_o = np.zeros(M)
    store_lJ_o = np.zeros(M)
    c_h = -T * n / 2 * np.log(2 * np.pi) + np.sum(np.log(CKh_hat.ab[0]))
    nuh, Sh = np.asarray(Hyper['nuh'], dtype=float), np.asarray(Hyper['Sh'], dtype=float)
    t_idx = np.arange(T)
    for isim in range(M):
        if is_kappafixed:
            kappa = np.r_[store_kappa[0, :2], big_kappa4[isim]]
        elif is_kappasym:
            kappa = np.r_[big_kappa[isim, 0], big_kappa[isim, 0], big_kappa[isim, 1]]
        else:
            kappa = big_kappa[isim]
        alp0, Valp = minn(p, kappa[0], kappa[1], kappa3, Y0, Y, 4)[:2]
        beta0, Vbeta = impact_B0(Y0, Y, kappa[2])
        mu = big_mu[isim]
        phi = big_phi[isim]
        longh = h_hat + CKh_hat.solve_t(r.randn(T * n))
        h = longh.reshape((T, n), order='F')
        beta = betahat + betaCcov @ r.randn(k_beta)
        B0[B0_id] = beta
        po = big_po[isim]
        o_idx = ngrid - np.sum(r.rand(T)[:, None] < cumsumo_hat, axis=1)    # counted from 0
        o = o_grid[o_idx]
            # defect 3: under bugcompat the outlier Jacobian is left out of c1
        lJ_o = 0.0 if bugcompat else -n * np.sum(np.log(o))

        c1 = -n * T / 2 * np.log(2 * np.pi) - .5 * np.sum(h) - .5 * np.sum(np.log(Valp)) + lJ_o
        iValp = 1 / Valp
        if blocks:
            XWX, XWy, yWy = kron_gram(X, B0, np.exp(-h) / o[:, None]**2, Y @ B0.T)
        else:
            diag_sqrt_D = (np.exp(h / 2) * o[:, None]).ravel(order='F')
            ytilde = (Y @ B0.T).ravel(order='F') / diag_sqrt_D
            Xtilde = np.kron(B0, X) / diag_sqrt_D[:, None]
            XWX, XWy, yWy = Xtilde.T @ Xtilde, Xtilde.T @ ytilde, np.sum(ytilde**2)
        Kalp = XWX.copy()
        Kalp[np.diag_indices(n * k)] += iValp
        CKalp = np.linalg.cholesky(Kalp)
        tmpc = solve_triangular(CKalp, iValp * alp0 + XWy, lower=True)
        lclike = c1 - np.sum(np.log(np.diag(CKalp))) \
            - .5 * (yWy + np.sum(alp0**2 / Valp) - tmpc @ tmpc)
        lh_pri = 0.0
        for ii in range(n):
            q = _hish_quad(h[:, ii] - mu[ii], phi[ii])
            lh_pri += -T / 2 * np.log(2 * np.pi) + .5 * np.log(1 - phi[ii]**2) \
                + nuh[ii] * np.log(Sh[ii]) - gammaln(nuh[ii]) + gammaln(nuh[ii] + T / 2) \
                - (nuh[ii] + T / 2) * np.log(Sh[ii] + .5 * q)
        dh = longh - h_hat
        lh_g = c_h - .5 * dh @ (Kh_hat @ dh)
        tmpbeta = solve_triangular(betaCcov, beta - betahat, lower=True)
        lr_beta = -.5 * np.sum(np.log(Vbeta)) - .5 * np.sum((beta - beta0)**2 / Vbeta) \
            + np.sum(np.log(np.diag(betaCcov))) + .5 * (tmpbeta @ tmpbeta)
            # defects 1-2: prior mass over n_pri atoms, and how o_hat is indexed
        o_lpri = np.log(np.r_[1 - po, np.repeat(po / n_pri, n_pri)])
        if bugcompat:              # linear index: column 1 of o_hat for T >= 32
            o_hat_at = o_hat.ravel(order='F')[o_idx]
        else:                      # (t, o_idx(t))
            o_hat_at = o_hat[t_idx, o_idx]
        lr_o = np.sum(o_lpri[o_idx] - np.log(o_hat_at))
        xk = kappa[2:3] if is_kappafixed else (kappa[1:3] if is_kappasym else kappa)
        store_w[isim] = lclike + lh_pri - lh_g + lr_beta + lr_o + prior(mu, phi, xk, po) \
            - gIS(mu, phi, xk, po)
        store_lr_o[isim] = lr_o
        store_lJ_o[isim] = lJ_o
    shortw = store_w.reshape((M // 50, 50), order='F')
    maxw = shortw.max(axis=0)
    bigml = np.log(np.mean(np.exp(shortw - maxw), axis=0)) + maxw
    lml = np.mean(bigml)
    lmlstd = np.std(bigml, ddof=1) / np.sqrt(50)

    out = Struct(store_w=store_w, bigml=bigml, M=M, bugcompat=bugcompat, store_lr_o=store_lr_o,
                 store_lJ_o=store_lJ_o, o_hat=o_hat, po_hat=po_hat, ngrid_atoms=ngrid,
                 n_prior_atoms=n_pri, h_hat=h_hat, Kh_hat=Kh_hat, betahat=betahat,
                 betacov=betacov, muhat=muhat, muvar=muvar, phihat=phihat, phivar=phivar,
                 ckappa_hat=ckappa_hat)
    return lml, lmlstd, out
