import numpy as np

from .._rng import as_rng
from .._struct import Struct
from ..priors import minnesota_C, resid_var_ar4, vtheta
from ..samplers import alp_tri_cs, eq_tri_cs, eq_var_oi, horseshoe_kappa_psi
from ..structural import b0_row_sampler, construct_Sigt
from ..sv import ksc_ar1_mean, sv0_params, sv_params
from ..util import build_lags


def var_sv(Y0, Y, p, model='OI', nsim=30000, burnin=5000, rng=None, draws=False,
           phi_proposal='truncated'):
    """bvt.models.var_sv - posterior sampler for a VAR with stochastic volatility,
    under the Cholesky or the order-invariant specification of the error
    covariance matrix, with the prior of Chan, Koop and Yu (2024).

      res = bvt.models.var_sv(Y0, Y, p)
      res = bvt.models.var_sv(Y0, Y, p, model='CS', nsim=1000, rng=1)

      Y0 : initial conditions, at least max(p,4) rows, same columns as Y
      Y  : T x n estimation sample, n >= 2, each series stationary, no NaN or Inf
      p  : lag length

      model  : 'OI' (default), Sigma_t^{-1} = B0' D_t^{-1} B0 with B0 unrestricted
               and zero-mean log-volatilities; or 'CS', the same with B0 unit
               lower triangular and log-volatility means estimated
      nsim   : posterior draws kept, default 30000
      burnin : draws discarded first, default 5000
      rng    : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      draws  : True also returns res.draws, one row per draw: kappa (nsim x 2),
               A (nsim x k*n, rows A.ravel(order='F')), impact (nsim x n^2), phi, sig2
               and h_T (nsim x n each, h_T at the end of the sample), mu (nsim x n)
               under 'CS'; default False. Full log-volatility paths are not kept
      phi_proposal : under 'OI', the Metropolis-Hastings candidate for phi,
               passed to bvt.sv.sv0_params: 'truncated' (default) or
               'untruncated', the step of the Chan, Koop and Yu (2024) package

      res.Sig_mean    : T x n x n posterior mean of Sigma_t
      res.A_mean      : k x n posterior mean of the coefficients, intercept first
                        (k = 1 + n*p)
      res.h_mean      : T x n posterior mean of the log-volatilities
      res.impact_mean : n x n posterior mean of B0 ('OI') or of the unit lower
                        triangular matrix ('CS')
      res.kappa_mean  : posterior means of the own- and cross-lag shrinkage
      plus the settings: model, nsim, burnin, seed, p

    The prior constants and the chain initialization are those of the Chan, Koop
    and Yu (2024) package.

    See:
    Cogley, T. and Sargent, T.J. (2005). Drifts and Volatilities: Monetary
    Policies and Outcomes in the Post WWII US, Review of Economic Dynamics,
    8(2): 262-302.
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    model = str(model).upper()
    phi_proposal = str(phi_proposal).lower()
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    if model not in ('OI', 'CS'):
        raise ValueError(f"model must be 'OI' or 'CS'; got '{model}'")
    if n < 2:
        raise ValueError('Y must have at least two columns (variables)')
    if Y0.shape[1] != n:
        raise ValueError(f'Y0 has {Y0.shape[1]} columns and Y has {n}')
    if Y0.shape[0] < max(p, 4):
        raise ValueError(f'Y0 has {Y0.shape[0]} rows; the lags and the AR(4) prior scaling need '
                         f'at least {max(p, 4)}')
    if not (np.all(np.isfinite(Y0)) and np.all(np.isfinite(Y))):
        raise ValueError('Y0 and Y contain NaN or Inf; the sampler has no step for missing values')
    if not (np.ndim(nsim) == 0 and nsim >= 1 and nsim == int(nsim) and np.ndim(burnin) == 0
            and burnin >= 0 and burnin == int(burnin)):
        raise ValueError('nsim must be a positive integer and burnin a nonnegative integer')
    if phi_proposal not in ('truncated', 'untruncated'):
        raise ValueError(f"phi_proposal must be 'truncated' or 'untruncated'; got '{phi_proposal}'")
    nsim, burnin, p = int(nsim), int(burnin), int(p)
    seed = int(rng) if isinstance(rng, (int, np.integer)) else None
    r = as_rng(rng)

    is_oi = model == 'OI'
    k = 1 + n * p
    _, X = build_lags(np.vstack([Y0[-p:], Y]), p)

        # constants of the package's preset.m
    ls_ridge = .01                             # chain-init least squares
    sv_offset = .0001                          # ystar = log(E**2 + sv_offset)
    phi_init_bnd = .99                         # chain-init bound on phi
    kappa = np.array([.1, .1, np.nan, 100]) if is_oi else np.array([.1, .1, 1, 100])

        # Minnesota second moments, built from AR(4) residual variances
    sig2_ar = resid_var_ar4(Y0, Y)
    C, idx_kappa1, idx_kappa2 = minnesota_C(n, p, sig2_ar)

    Hyper = Struct(nuh=3 * np.ones(n), phi0=.95 * np.ones(n), Vphi=.05**2 * np.ones(n),
                   mu0=np.zeros(n), Vmu=100 * np.ones(n), B0=np.eye(n), VB0=np.ones((n, n)),
                   beta0=np.zeros(n * n * p + n), Valp=np.ones(n * (n - 1) // 2))
    Hyper.Sh = .05 * (Hyper.nuh - 1)

        # chain init
    A = np.linalg.solve(X.T @ X + ls_ridge * np.eye(k), X.T @ Y)
    U = Y - X @ A
    Sig_hat = U.T @ U / T
    h = np.tile(np.log(np.diag(Sig_hat)), (T, 1))
    sig2 = 1 / r.gamrnd(Hyper.nuh, 1 / Hyper.Sh)
    phi = np.minimum(Hyper.phi0 + np.sqrt(Hyper.Vphi) * r.randn(n), phi_init_bnd)
    z_psi1 = 1 / r.gamrnd(.5, 1, n * p)
    z_psi2 = 1 / r.gamrnd(.5, 1, (n - 1) * n * p)
    z_kappa = 1 / r.gamrnd(.5, 1, 2)
    Psi = np.ones(k * n)
    Psi[idx_kappa1] = 1 / r.gamrnd(.5, z_psi1)
    Psi[idx_kappa2] = 1 / r.gamrnd(.5, z_psi2)
    if is_oi:
        B0 = np.diag(1 / np.sqrt(np.diag(Sig_hat)))   # full matrix, updated row by row
    else:
        A_id = np.tril_indices(n, -1)
        Atri = np.eye(n)                                # unit lower triangular impact matrix
        B = A.T.copy()
        XB = X @ B.T
        mu = np.array([np.mean(np.log(U[:, ii]**2)) for ii in range(n)])

    Sig_sum = np.zeros((T, n, n))
    A_sum = np.zeros((k, n))
    h_sum = np.zeros((T, n))
    imp_sum = np.zeros((n, n))
    store_kappa = np.zeros((nsim, 2))
    if draws:
        D = Struct(A=np.zeros((nsim, k * n)), impact=np.zeros((nsim, n * n)),
                   phi=np.zeros((nsim, n)), sig2=np.zeros((nsim, n)), h_T=np.zeros((nsim, n)))
        if not is_oi:
            D.mu = np.zeros((nsim, n))
    for isim in range(1, nsim + burnin + 1):
            # Vbeta scales with sig2, which holds the log-volatility state variances
            # at this point, as in the published samplers
        _, tmpdV = vtheta(idx_kappa1, idx_kappa2, kappa, C * Psi, sig2)

        if is_oi:
            B0 = b0_row_sampler(Y - X @ A, h, B0, Hyper.B0, Hyper.VB0, rng=r)
            A = eq_var_oi(Y, X, B0, h, A, tmpdV, rng=r)
            theta = A.ravel(order='F')
            E = (Y - X @ A) @ B0.T                     # structural innovations
        else:
            B, XB = eq_tri_cs(Y, X, XB, B, Atri, h, tmpdV, Hyper.beta0, rng=r)
            theta = B.ravel()                          # B' stacked column by column
            E = Y - XB
            Atri[A_id] = alp_tri_cs(E, h, Hyper.Valp, rng=r)
            E = E @ Atri.T                             # structural innovations

        for ii in range(n):
            ystar = np.log(E[:, ii]**2 + sv_offset)
            h[:, ii] = ksc_ar1_mean(ystar, h[:, ii], 0 if is_oi else mu[ii], phi[ii], sig2[ii],
                                    rng=r)[0]
        if is_oi:
            phi, sig2, _ = sv0_params(h, phi, Hyper, None, proposal=phi_proposal, rng=r)
        else:
            mu, phi, sig2, _ = sv_params(h, mu, phi, Hyper, rng=r)

        psi1, psi2, z_psi1, z_psi2, kappa, z_kappa = horseshoe_kappa_psi(
            theta, idx_kappa1, idx_kappa2, C, kappa, z_psi1, z_psi2, z_kappa, rng=r)
        Psi[idx_kappa1] = psi1
        Psi[idx_kappa2] = psi2

        if isim > burnin:
            i = isim - burnin - 1
            store_kappa[i] = kappa[:2]
            h_sum += h
            if is_oi:
                Sig_sum += construct_Sigt(h, B0)
                A_sum += A
                imp_sum += B0
            else:
                Sig_sum += construct_Sigt(h, Atri)
                A_sum += B.T
                imp_sum += Atri
            if draws:
                if is_oi:
                    D.A[i] = A.ravel(order='F')
                    D.impact[i] = B0.ravel(order='F')
                else:
                    D.A[i] = B.ravel()
                    D.impact[i] = Atri.ravel(order='F')
                    D.mu[i] = mu
                D.phi[i] = phi
                D.sig2[i] = sig2
                D.h_T[i] = h[-1]

    res = Struct(Sig_mean=Sig_sum / nsim, A_mean=A_sum / nsim, h_mean=h_sum / nsim,
                 impact_mean=imp_sum / nsim, kappa_mean=store_kappa.mean(axis=0), model=model,
                 nsim=nsim, burnin=burnin, seed=seed, p=p)
    if draws:
        D.kappa = store_kappa
        res.draws = D
    return res
