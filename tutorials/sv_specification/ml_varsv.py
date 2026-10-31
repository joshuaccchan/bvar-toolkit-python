"""ml_varsv - estimation and log marginal likelihoods of the five VARs of Chan (2023)

The Python version of the driver replications/chan2023_joe_mlvarsv/run_all.m and run_ml.m of
bvar-toolkit, for the volatility-specification tutorial: a homoskedastic natural-conjugate VAR
(VAR-NCP), common stochastic volatility (VAR-CSV), Cholesky stochastic volatility (VAR-SV),
factor stochastic volatility (VAR-FSV) and Cholesky stochastic volatility with an outlier
component (VAR-SVO), with the priors, chain initializations and sweep orders of that package.
run_ml estimates a model and then computes its log marginal likelihood with the matching
bvt.ml estimator on the same random stream; VAR-NCP has it in closed form.

The shrinkage hyperparameters are drawn from generalized inverse Gaussian distributions by
SciPy's geninvgauss; the package uses gigrnd, an implementation of the algorithm of Devroye
(2014).

The first n0 = 8 rows of the data are the initial conditions, and the VAR has p = 4 lags.

See:
Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large Bayesian VARs,
Journal of Econometrics, 235(2), 1419-1446.
Devroye, L. (2014). Random variate generation for the generalized inverse Gaussian
distribution, Statistics and Computing, 24(2), 239-246.
"""

import numpy as np
from scipy.linalg import solve_triangular

import bvar_toolkit as bvt

# the constants of the package's preset.m
PRESET = dict(n0=8, p=4, r=2, kappa=.2**2, kappa1=.2**2, kappa2=(.2**2)**2, kappa3=100,
              kappa4=.2**2, sv_offset=.0001, phi_mh_bnd=.998, nuh=3, Sh_factor=.1, mu0=0,
              Vmu=100, phi0=.98, Vphi=.05**2, csv_c0=[1, 1 / .2**2], forced_accept_before=20,
              sv_c0=[[1, 1 / .2**2], [1, 1 / .2**2], [1, 1]],
              fsv_c0=[[1, 1 / .2**2], [1, 1 / .2**2]],
              fsv_l0=0, fsv_Vl=1, fsv_A_init_VA=[100, .1], fsv_L_init_lower=1,
              fsv_phi_init=[.9, .09], fsv_sig2_init=[.05, .05], svo_p0a=10 / 4,
              svo_p0b=(1 - 1 / 16) * 40, svo_ngrid=31, svo_o_grid_range=[2, 20], svo_po_init=1 / 16,
              svo_o_init=1, M=10000, flag_marg=2)
MODELS = ['VAR-NCP', 'VAR-CSV', 'VAR-SV', 'VAR-FSV', 'VAR-SVO']


def run_all(model, data, is_kappafixed=False, is_kappasym=False, nsim=10000, burnin=1000,
            rng=None, r=2):
    """Estimate one of the five models on data, whose first eight rows are the initial
    conditions; returns a Struct with the stored draws, the posterior means and what the
    marginal likelihood needs."""
    pr = PRESET
    imodel = MODELS.index(model) + 1
    rg = bvt._rng.as_rng(rng)
    data = np.asarray(data, dtype=float)
    p = pr['p']
    Y0, Y = data[:pr['n0']], data[pr['n0']:]
    T, n = Y.shape
    _, X = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
    k_beta = n * (n - 1) // 2
    k_alp = n * n * p + n
    k = k_alp // n
    kappa, kappa1, kappa3, kappa4 = pr['kappa'], pr['kappa1'], pr['kappa3'], pr['kappa4']
    kappa2 = kappa1 if is_kappasym else pr['kappa2']
    sig2_hat = None
    Hyper = bvt.Struct()
    if imodel in (1, 2):
        Hyper.A0, Hyper.VA, Hyper.nu0, Hyper.S0, _ = bvt.priors.niw(p, [kappa, kappa3], Y0, Y,
                                                                    'mlvarsv_ncp')
        if imodel == 2:
            Hyper.c0 = np.array(pr['csv_c0'])
            Hyper.nuh = pr['nuh']
            Hyper.Sh = pr['Sh_factor'] * (Hyper.nuh - 1)
            Hyper.phi0, Hyper.Vphi = pr['phi0'], pr['Vphi']
    else:
        Hyper.alp0, Hyper.Valp, sig2_hat, _ = bvt.priors.minn(p, kappa1, kappa2, kappa3, Y0, Y, 4)
        m = n + r if imodel == 4 else n
        Hyper.nuh = pr['nuh'] * np.ones(m)
        Hyper.Sh = pr['Sh_factor'] * (Hyper.nuh - 1)
        Hyper.mu0, Hyper.Vmu = pr['mu0'] * np.ones(m), pr['Vmu'] * np.ones(m)
        Hyper.phi0, Hyper.Vphi = pr['phi0'] * np.ones(m), pr['Vphi'] * np.ones(m)
        if imodel == 4:
            Hyper.c0 = np.array(pr['fsv_c0'])
            Hyper.l0, Hyper.Vl = pr['fsv_l0'], pr['fsv_Vl']
        else:
            Hyper.beta0, Hyper.Vbeta = bvt.priors.impact_B0(Y0, Y, kappa4)
            Hyper.c0 = np.array(pr['sv_c0'])
            if imodel == 5:
                Hyper.p0a, Hyper.p0b = pr['svo_p0a'], pr['svo_p0b']
    dims = dict(T=T, n=n, p=p, r=r, k=k, k_alp=k_alp, k_beta=k_beta)
    if imodel == 1:
        res = _ncp(Y, X, dims, Hyper)
    elif imodel == 2:
        res = _mcmc_csv(Y, X, Y0, dims, Hyper, kappa, kappa3, is_kappafixed, nsim, burnin, rg)
    elif imodel == 3:
        res = _mcmc_arsv(Y, X, Y0, dims, Hyper, sig2_hat, kappa1, kappa2, kappa3, kappa4,
                         is_kappafixed, is_kappasym, nsim, burnin, rg, outliers=False)
    elif imodel == 4:
        res = _mcmc_fsv(Y, X, Y0, dims, Hyper, sig2_hat, kappa1, kappa2, kappa3, is_kappafixed,
                        is_kappasym, nsim, burnin, rg)
    else:
        res = _mcmc_arsv(Y, X, Y0, dims, Hyper, sig2_hat, kappa1, kappa2, kappa3, kappa4,
                         is_kappafixed, is_kappasym, nsim, burnin, rg, outliers=True)
    res.update(model=model, model_num=imodel, is_kappafixed=is_kappafixed,
               is_kappasym=is_kappasym, nsim=nsim, burnin=burnin, T=T, n=n, p=p, r=r, k=k,
               Y=Y, X=X, Y0=Y0, Hyper=Hyper)
    return res


def run_ml(model, data, is_kappafixed=False, is_kappasym=False, nsim=10000, burnin=1000,
           rng=None, r=2, M=None, flag_marg=None, bugcompat=False, gram='full'):
    """Estimate one of the five models and compute its log marginal likelihood on the same
    random stream; returns the estimation output with lml, lmlstd (None for VAR-NCP, whose
    log marginal likelihood is in closed form) and ml, the estimator's detail."""
    rg = bvt._rng.as_rng(rng)
    out = run_all(model, data, is_kappafixed, is_kappasym, nsim, burnin, rg, r)
    M = PRESET['M'] if M is None else M
    flag_marg = PRESET['flag_marg'] if flag_marg is None else flag_marg
    a = (out.X, out.Y, out.Y0, M, out.Hyper)
    kfix, ksym = is_kappafixed, is_kappasym
    if out.model_num == 1:
        lml, lmlstd, detail = out.lml, None, None
    elif out.model_num == 2:
        lml, lmlstd, detail = bvt.ml.mlvarsv_csv(*a, out.store_h, out.store_hpara,
                                                 out.store_kappa, kfix, rng=rg)
    elif out.model_num == 3:
        lml, lmlstd, detail = bvt.ml.mlvarsv_arsv_redu(*a, flag_marg, out.store_h, out.store_beta,
                                                       out.store_hpara, out.store_kappa, kfix,
                                                       ksym, gram=gram, rng=rg)
    elif out.model_num == 4:
        lml, lmlstd, detail = bvt.ml.mlvarsv_fsv(*a, flag_marg, out.store_h, out.store_hpara,
                                                 out.store_l, out.store_kappa, kfix, ksym,
                                                 gram=gram, rng=rg)
    else:
        lml, lmlstd, detail = bvt.ml.mlvarsv_arsvo_redu(*a, flag_marg, out.store_h,
                                                        out.store_beta, out.store_hpara,
                                                        out.store_kappa, out.store_o,
                                                        out.store_po, out.o_grid, kfix, ksym,
                                                        bugcompat=bugcompat, gram=gram, rng=rg)
    out.update(lml=lml, lmlstd=lmlstd, ml=detail, ml_M=M, ml_flag_marg=flag_marg)
    return out


def _ncp(Y, X, dims, Hyper):
    """The homoskedastic VAR: posterior and log marginal likelihood in closed form."""
    T, n, k = dims['T'], dims['n'], dims['k']
    XX = X.T @ X
    A_tilde = np.linalg.solve(XX, X.T @ Y)
    K_A = XX.copy()
    K_A[np.diag_indices(k)] += 1 / Hyper.VA
    CK_A = np.linalg.cholesky(K_A)
    A_hat = solve_triangular(CK_A, solve_triangular(CK_A, Hyper.A0 / Hyper.VA[:, None]
                                                    + XX @ A_tilde, lower=True), lower=True,
                             trans='T')
    S_hat = Hyper.S0 + Hyper.A0.T @ (Hyper.A0 / Hyper.VA[:, None]) + Y.T @ Y - A_hat.T @ K_A @ A_hat
    S_hat = (S_hat + S_hat.T) / 2
    lml = (-n * T / 2 * np.log(np.pi)
           - n / 2 * (np.sum(np.log(Hyper.VA)) + 2 * np.sum(np.log(np.diag(CK_A))))
           + Hyper.nu0 / 2 * bvt.util.ldet(Hyper.S0) - (Hyper.nu0 + T) / 2 * bvt.util.ldet(S_hat)
           + bvt.util.mgammaln(n, (Hyper.nu0 + T) / 2) - bvt.util.mgammaln(n, Hyper.nu0 / 2))
    return bvt.Struct(A_tilde=A_tilde, K_A=K_A, A_hat=A_hat, S_hat=S_hat, lml=lml)


def _mcmc_csv(Y, X, Y0, dims, Hyper, kappa, kappa3, is_kappafixed, nsim, burnin, rg):
    """VAR-CSV. Order per sweep: (Sig, A), h, (phi, sig2), kappa."""
    T, n, p, k = dims['T'], dims['n'], dims['p'], dims['k']
    store_Sig = np.zeros((nsim, n * (n + 1) // 2))
    store_a = np.zeros((nsim, k * n))
    store_h = np.zeros((nsim, T))
    store_hpara = np.zeros((nsim, 2))
    store_kappa = np.zeros(nsim)

        # initialize the chain
    sig2 = 1 / rg.gamrnd(Hyper.nuh, 1 / Hyper.Sh)
    phi = Hyper.phi0
    A0, VA = bvt.priors.niw(p, [kappa, kappa3], Y0, Y, 'mlvarsv_ncp')[:2]
    C = VA / kappa                         # the constant part of VA
    K_A = X.T @ X
    K_A[np.diag_indices(k)] += 1 / VA
    A = np.linalg.solve(K_A, A0 / VA[:, None] + X.T @ Y)
    Sig = (Hyper.S0 + A0.T @ (A0 / VA[:, None]) + Y.T @ Y - A.T @ K_A @ A) / T
    U = Y - X @ A
    s2 = np.sum(solve_triangular(np.linalg.cholesky(Sig), U.T, lower=True)**2, axis=0)
    h = bvt.sv.csv_armh(s2, phi, sig2, np.zeros(T), n, True, rng=rg)[0]
    count_h = 0
    count_phi = 0.0

    for isim in range(1, nsim + burnin + 1):
            # sample Sig and A
        VA = bvt.priors.niw(p, [kappa, kappa3], Y0, Y, 'mlvarsv_ncp')[1]
        XiOh = X.T * np.exp(-h)
        K_A = XiOh @ X
        K_A[np.diag_indices(k)] += 1 / VA
        CK_A = np.linalg.cholesky(K_A)
        A_hat = solve_triangular(CK_A, solve_triangular(CK_A, A0 / VA[:, None] + XiOh @ Y,
                                                        lower=True), lower=True, trans='T')
        S_hat = Hyper.S0 + A0.T @ (A0 / VA[:, None]) + (Y.T * np.exp(-h)) @ Y \
            - A_hat.T @ K_A @ A_hat
        S_hat = (S_hat + S_hat.T) / 2      # adjust for rounding errors
        Sig = rg.iwishrnd(S_hat, Hyper.nu0 + T)
        CSig = np.linalg.cholesky(Sig)
        A = A_hat + solve_triangular(CK_A, rg.randn(k, n), lower=True, trans='T') @ CSig.T

            # sample h
        U = Y - X @ A
        s2 = np.sum(solve_triangular(CSig, U.T, lower=True)**2, axis=0)
        if isim < PRESET['forced_accept_before']:
            h = bvt.sv.csv_armh(s2, phi, sig2, h, n, True, rng=rg)[0]
            flag = 0
        else:
            h, flag = bvt.sv.csv_armh(s2, phi, sig2, h, n, rng=rg)
        count_h += flag

            # sample phi and sig2; a mean of 0 skips the mean block
        _, phi, sig2, flag_phi = bvt.sv.sv_params(h, 0, phi, Hyper, PRESET['phi_mh_bnd'], rng=rg)
        phi, sig2 = phi[0], sig2[0]
        count_phi += flag_phi[0]

            # sample kappa
        if not is_kappafixed:
            D = A - A0
            Q = np.sum(D * solve_triangular(CSig, solve_triangular(CSig, D.T, lower=True),
                                            lower=True, trans='T').T, axis=1)
            tmpc = np.sum(Q[1:] / C[1:])
            kappa = rg.gigrnd(Hyper.c0[0] - n**2 * p / 2, 2 * Hyper.c0[1], tmpc)

        if isim > burnin:
            i = isim - burnin - 1
            store_a[i] = A.ravel(order='F')
            store_Sig[i] = bvt.util.vech(Sig)
            store_h[i] = h
            store_hpara[i] = [phi, sig2]
            store_kappa[i] = kappa
    return bvt.Struct(store_Sig=store_Sig, store_a=store_a, store_h=store_h,
                      store_hpara=store_hpara, store_kappa=store_kappa, count_h=count_h,
                      count_phi=count_phi, h_mean=store_h.mean(axis=0),
                      hpara_mean=store_hpara.mean(axis=0), kappa_mean=store_kappa.mean(),
                      CSV_std_mean=np.exp(store_h / 2).mean(axis=0))


def _draw_kappas(alp, beta, C_alp, C_beta, idx1, idx2, Hyper, kappas, is_kappafixed,
                 is_kappasym, dims, rg, impact=True):
    """The GIG draws of kappa4 (with an impact matrix) and then of kappa1 and kappa2."""
    n, p = dims['n'], dims['p']
    kappa1, kappa2, kappa4 = kappas
    c0 = Hyper.c0
    if impact:
        tmpc4 = np.sum(beta**2 / C_beta)
        kappa4 = rg.gigrnd(c0[2][0] - n * (n - 1) / 4, 2 * c0[2][1], tmpc4)
    if is_kappafixed:
        pass
    elif is_kappasym:
        tmpc1 = np.sum(alp[idx1]**2 / C_alp[idx1]) + np.sum(alp[idx2]**2 / C_alp[idx2])
        kappa1 = rg.gigrnd(c0[0][0] - n**2 * p / 2, 2 * c0[0][1], tmpc1)
        kappa2 = kappa1
    else:
        tmpc1 = np.sum(alp[idx1]**2 / C_alp[idx1])
        tmpc2 = np.sum(alp[idx2]**2 / C_alp[idx2])
        kappa1 = rg.gigrnd(c0[0][0] - n * p / 2, 2 * c0[0][1], tmpc1)
        kappa2 = rg.gigrnd(c0[1][0] - (n - 1) * n * p / 2, 2 * c0[1][1], tmpc2)
    return kappa1, kappa2, kappa4


def _mcmc_arsv(Y, X, Y0, dims, Hyper, sig2_hat, kappa1, kappa2, kappa3, kappa4, is_kappafixed,
               is_kappasym, nsim, burnin, rg, outliers):
    """VAR-SV, and VAR-SVO when outliers is True. Order per sweep: alp, beta, h,
    (mu, phi, sig2), kappa4, kappa1, kappa2, and under VAR-SVO then (o, po)."""
    T, n, p, k, k_alp, k_beta = (dims[s] for s in ('T', 'n', 'p', 'k', 'k_alp', 'k_beta'))
    store_alp = np.zeros((nsim, k_alp))
    store_beta = np.zeros((nsim, k_beta))
    store_h = np.zeros((nsim, T, n))
    store_hpara = np.zeros((nsim, 3 * n))
    store_kappa = np.zeros((nsim, 3))
    count_phi = np.zeros(n)
    if outliers:
        ngrid = PRESET['svo_ngrid']
        lo, hi = PRESET['svo_o_grid_range']
        o_grid = np.r_[1, np.linspace(lo, hi, ngrid)]
        store_o = np.zeros((nsim, T))
        store_po = np.zeros(nsim)
        po = PRESET['svo_po_init']
        o = PRESET['svo_o_init'] * np.ones(T)
    else:
        o = None

        # initialize the Markov chain
    mu = np.log(sig2_hat)
    sig2 = 1 / rg.gamrnd(Hyper.nuh, 1 / Hyper.Sh)
    phi = Hyper.phi0.copy()
    A = np.zeros((k, n))
    h = np.zeros((T, n))
    XX = X.T @ X
    for ii in range(n):
        K = XX.copy()
        K[np.diag_indices(k)] += 1 / Hyper.Valp[ii * k:(ii + 1) * k]
        A[:, ii] = np.linalg.solve(K, X.T @ Y[:, ii])
        s2i = (Y[:, ii] - X @ A[:, ii])**2
        mu[ii] = np.mean(np.log(s2i))
        h[:, ii] = bvt.sv.init_approx1N(s2i, mu[ii], phi[ii], sig2[ii])
    B0_id = np.tril_indices(n, -1)
    B0 = np.eye(n)
    C_alp, idx1, idx2 = bvt.priors.minnesota_C(n, p, sig2_hat)
    C_beta = Hyper.Vbeta / kappa4          # the constant part of Vbeta

    for isim in range(1, nsim + burnin + 1):
            # sample alp
        Valp = bvt.priors.minn(p, kappa1, kappa2, kappa3, Y0, Y, 4)[1]
        A = bvt.samplers.eq_var_redu_tri(Y, X, B0, h, A, Valp, Hyper.alp0, o, rng=rg)
        alp = A.ravel(order='F')

            # sample beta
        Vbeta = bvt.priors.impact_B0(Y0, Y, kappa4)[1]
        E = Y - X @ A
        beta = bvt.samplers.alp_tri_cs(E, h, Vbeta, o, rng=rg)
        B0[B0_id] = beta

            # sample h
        B0E = (E if o is None else E / o[:, None]) @ B0.T
        for ii in range(n):
            ystar = np.log(B0E[:, ii]**2 + PRESET['sv_offset'])
            h[:, ii] = bvt.sv.ksc_ar1_mean(ystar, h[:, ii], mu[ii], phi[ii], sig2[ii], rng=rg)[0]

            # sample mu, phi and sig2
        mu, phi, sig2, flag_phi = bvt.sv.sv_params(h, mu, phi, Hyper, PRESET['phi_mh_bnd'],
                                                   rng=rg)
        count_phi += flag_phi

            # sample kappa4, kappa1 and kappa2
        kappa1, kappa2, kappa4 = _draw_kappas(alp, beta, C_alp, C_beta, idx1, idx2, Hyper,
                                              (kappa1, kappa2, kappa4), is_kappafixed,
                                              is_kappasym, dims, rg)

        if outliers:            # sample o and po
            o, po = bvt.sv.svo_outlier(Y, X, A, B0, h, o_grid, po, Hyper.p0a, Hyper.p0b, rng=rg)

        if isim > burnin:
            i = isim - burnin - 1
            store_alp[i] = A.ravel(order='F')
            store_beta[i] = beta
            store_h[i] = h
            store_hpara[i] = np.r_[mu, phi, sig2]
            store_kappa[i] = [kappa1, kappa2, kappa4]
            if outliers:
                store_o[i] = o
                store_po[i] = po
    res = bvt.Struct(store_alp=store_alp, store_beta=store_beta, store_h=store_h,
                     store_hpara=store_hpara, store_kappa=store_kappa, count_phi=count_phi,
                     h_mean=store_h.mean(axis=0), hpara_mean=store_hpara.mean(axis=0),
                     kappa_mean=store_kappa.mean(axis=0))
    if outliers:
        res.update(store_o=store_o, store_po=store_po, o_mean=store_o.mean(axis=0),
                   po_mean=store_po.mean(), o_grid=o_grid)
    return res


def _mcmc_fsv(Y, X, Y0, dims, Hyper, sig2_hat, kappa1, kappa2, kappa3, is_kappafixed,
              is_kappasym, nsim, burnin, rg):
    """VAR-FSV. Order per sweep: F, (A, L), h, (mu, phi, sig2), kappa1, kappa2."""
    T, n, p, r, k = (dims[s] for s in ('T', 'n', 'p', 'r', 'k'))
    L_idx = np.nonzero(np.tril(np.ones((n, r)), -1).T)[::-1]   # free loadings, column by column
    store_l = np.zeros((nsim, L_idx[0].size))
    store_A = np.zeros((k, n))
    store_h = np.zeros((nsim, T, n + r))
    store_F = np.zeros((nsim, T, r))
    store_hpara = np.zeros((nsim, 3 * (n + r)))
    count_phi = np.zeros(n + r)
    store_kappa = np.zeros((nsim, 2))

        # initialize the Markov chain
    va = PRESET['fsv_A_init_VA']
    K = X.T @ X
    K[np.diag_indices(k)] += 1 / np.r_[va[0], va[1] * np.ones(k - 1)]
    A = np.linalg.solve(K, X.T @ Y)
    L = np.vstack([np.eye(r), PRESET['fsv_L_init_lower'] * np.ones((n - r, r))])
    varY = Y.var(axis=0, ddof=1)
    mu = np.r_[np.log(varY / 2), np.log(np.mean(varY)) * np.ones(r)]
    phi = PRESET['fsv_phi_init'][0] + PRESET['fsv_phi_init'][1] * rg.rand(n + r)
    sig2 = PRESET['fsv_sig2_init'][0] + PRESET['fsv_sig2_init'][1] * rg.rand(n + r)
    Uy = Y - X @ A
    h = np.hstack([np.zeros((T, n)), np.tile(mu[n:], (T, 1))])
    for ii in range(n):
        h[:, ii] = bvt.sv.init_approx1N(Uy[:, ii]**2, mu[ii], phi[ii], sig2[ii])
    C_alp, idx1, idx2 = bvt.priors.minnesota_C(n, p, sig2_hat)

    for isim in range(1, nsim + burnin + 1):
            # sample F
        F = bvt.samplers.factor_fsv(Y, X, A, L, h, rng=rg)

            # sample L and A
        Valp = bvt.priors.minn(p, kappa1, kappa2, kappa3, Y0, Y, 4)[1]
        A, L = bvt.samplers.eq_fsv_load(Y, X, F, h, A, L, Valp, Hyper.alp0, Hyper.Vl, Hyper.l0,
                                        rng=rg)
        alp = A.ravel(order='F')

            # sample h
        Uy = Y - X @ A - F @ L.T
        ystar = np.log(np.hstack([Uy, F])**2 + PRESET['sv_offset'])
        for jj in range(n + r):
            h[:, jj] = bvt.sv.ksc_ar1_mean(ystar[:, jj], h[:, jj], mu[jj], phi[jj], sig2[jj],
                                           rng=rg)[0]

            # sample mu, phi and sig2
        mu, phi, sig2, flag_phi = bvt.sv.sv_params(h, mu, phi, Hyper, PRESET['phi_mh_bnd'],
                                                   rng=rg)
        count_phi += flag_phi

            # sample kappa1 and kappa2
        kappa1, kappa2, _ = _draw_kappas(alp, None, C_alp, None, idx1, idx2, Hyper,
                                         (kappa1, kappa2, None), is_kappafixed, is_kappasym,
                                         dims, rg, impact=False)

        if isim > burnin:
            i = isim - burnin - 1
            store_l[i] = L[L_idx]
            store_A += A
            store_h[i] = h
            store_F[i] = F
            store_hpara[i] = np.r_[mu, phi, sig2]
            store_kappa[i] = [kappa1, kappa2]
    L_mean = np.vstack([np.eye(r), np.ones((n - r, r))])
    L_mean[L_idx] = store_l.mean(axis=0)
    return bvt.Struct(store_l=store_l, store_A=store_A, store_h=store_h, store_F=store_F,
                      store_hpara=store_hpara, store_kappa=store_kappa, count_phi=count_phi,
                      l_mean=store_l.mean(axis=0), L_mean=L_mean, A_mean=store_A / nsim,
                      h_mean=store_h.mean(axis=0), F_mean=store_F.mean(axis=0),
                      hpara_mean=store_hpara.mean(axis=0), kappa_mean=store_kappa.mean(axis=0))
