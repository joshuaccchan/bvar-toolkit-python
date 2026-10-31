import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

from .._rng import as_rng
from .._struct import Struct
from ..samplers import missing_var, var_coef_joint
from ..sv import csv_armh_block, sv0_params
from ..util import build_lags


def mfvar_csv(Y, p, M=None, z=None, sig2=None, kappa=(.04, .01), V0=100, m0=0, nsim=10000,
              burnin=1000, rng=None, c_reject=3, hblock=36, quantiles=(.05, .16, .5, .84, .95),
              draws=False):
    """bvt.models.mfvar_csv - posterior sampler for a VAR with common stochastic volatility and
    missing data, such as a mixed-frequency VAR, with the independent priors of Chan, Poon
    and Zhu (2023).

      res = bvt.models.mfvar_csv(Y, p)
      res = bvt.models.mfvar_csv(Y, p, M=M, z=z, sig2=s2, nsim=10000, rng=1)

      Y  : T x n data, NaN wherever a value is missing; the first p rows are the initial
           conditions. A mixed-frequency VAR has the quarterly series missing in every month
      p  : lag length

      M, z      : linear restrictions M @ y = z on y = vec(Y') = Y.ravel(), such as the
                  quarterly aggregation of bvt.util.mm_constraint; default none
      sig2      : n-vector of the variances s_i^2 that scale the Minnesota prior; a NaN
                  entry is replaced by the residual variance of an AR(4) fitted to the
                  observed values of that series, which a series missing in every period
                  does not have, so its entry must be given
      kappa     : (kappa_1, kappa_2), own-lag and other-lag shrinkage, default (.04, .01)
      V0, m0    : prior variance and mean of the missing initial conditions, defaults 100, 0
      nsim      : draws kept, default 10000;  burnin : draws discarded first, default 1000
      rng       : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      hblock    : block length of the h update, default 36
      c_reject  : envelope constant of the h update, default 3
      quantiles : probabilities of the reported quantiles, default (.05, .16, .5, .84, .95)
      draws     : True also returns res.draws: ym (nsim x Nm, float32), A (nsim x k*n),
                  Sig (nsim x n^2), h (nsim x T-p), phi and sigh2; default False

      res.Y_mean    : T x n data with each missing value replaced by its posterior mean
      res.Y_q       : T x n x len(quantiles) quantiles, equal to the data where observed
      res.A_mean    : k x n posterior mean of the VAR coefficients, intercept first, k = 1+n*p
      res.Sig_mean, res.h_mean (T-p), res.phi_mean, res.sigh2_mean : posterior means
      res.accept_rate : share of the blocks of h whose Metropolis step accepted, past the
                  first 20 sweeps, which take the proposal regardless
      res.max_resid : largest |M @ y - z| over the kept draws
      res.sig2  : the Minnesota scales used;  res.prior.V : k x n prior variances of A;
                  plus the settings and the rest of res.prior

    The model is y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + e_t with
    e_t ~ N(0, exp(h_t)*Sig) and h_t a zero-mean AR(1). Priors: the coefficients independent
    normal with mean zero and variances kappa_1/l^2 on own lags, kappa_2*s_i^2/(l^2*s_j^2)
    on lag l of variable j in equation i, and 100*s_i^2 on the intercepts;
    Sig ~ IW(n+3, I_n); sigh2 ~ IG(10, .004); phi ~ N(.98, .05^2) on |phi| < 1. Each sweep
    draws the missing values (bvt.samplers.missing_var), the coefficients jointly
    (bvt.samplers.var_coef_joint), Sig, h (bvt.sv.csv_armh_block) and (phi, sigh2)
    (bvt.sv.sv0_params, truncated candidate). The draws of h, here in blocks, and of
    (phi, sigh2) follow the algorithm of Chan (2020). Written for this toolkit.

    See:
    Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
    State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
    Section 4.1.
    Carriero, A., Clark, T.E. and Marcellino, M. (2016). Common Drifting Volatility in
    Large Bayesian VARs, Journal of Business and Economic Statistics, 34(3): 375-390.
    Chan, J.C.C. (2020). Large Bayesian VARs: A Flexible Kronecker Error Covariance
    Structure, Journal of Business and Economic Statistics, 38(1): 68-79.
    Schorfheide, F. and Song, D. (2015). Real-Time Forecasting with a Mixed-Frequency VAR,
    Journal of Business and Economic Statistics, 33(3): 366-380.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError('Y must have at least two columns (variables)')
    T, n = Y.shape
    if T <= p + 4:
        raise ValueError(f'Y has {T} rows; it needs more than p + 4')
    if np.any(np.isinf(Y)):
        raise ValueError('Y contains Inf; mark missing values with NaN')
    if not (np.ndim(nsim) == 0 and nsim >= 1 and nsim == int(nsim) and np.ndim(burnin) == 0
            and burnin >= 0 and burnin == int(burnin)):
        raise ValueError('nsim must be a positive integer and burnin a nonnegative integer')
    nsim, burnin = int(nsim), int(burnin)
    sig2 = np.full(n, np.nan) if sig2 is None else np.array(sig2, dtype=float).ravel()
    if sig2.size != n:
        raise ValueError('sig2 must have one entry per variable')
    for i in np.flatnonzero(np.isnan(sig2)):
        sig2[i] = _ar4_resid_var(Y[:, i])
    if np.any(np.isnan(sig2)):
        raise ValueError(f'sig2 must be given for series {list(np.flatnonzero(np.isnan(sig2)))}, '
                         'which have too few observed values for an AR(4)')
    seed = int(rng) if isinstance(rng, (int, np.integer)) else None
    r = as_rng(rng)
    probs = np.asarray(quantiles, dtype=float).ravel()

    # priors
    k = 1 + n * p
    V = np.zeros((k, n))                               # prior variances, one column per equation
    for i in range(n):
        V[0, i] = 100 * sig2[i]
        for lag in range(1, p + 1):
            for j in range(n):
                if i == j:
                    V[1 + (lag - 1) * n + j, i] = kappa[0] / lag**2
                else:
                    V[1 + (lag - 1) * n + j, i] = kappa[1] * sig2[i] / (lag**2 * sig2[j])
    iVb = sp.diags_array(1 / V.ravel(order='F'))
    nu0 = n + 3
    S0 = np.eye(n)
    # phi0 and Vphi, which Chan, Poon and Zhu (2023) do not give, are those of bvt.models.var_csv
    Hyper = Struct(nuh=10, Sh=.004, phi0=.98, Vphi=.05**2)
    Te = T - p

    # starting values: coefficients at the observed means, the missing values at their
    # conditional mean under those coefficients
    obs = ~np.isnan(Y)
    cnt = obs.sum(axis=0)
    mu = np.where(cnt > 0, np.where(obs, Y, 0).sum(axis=0) / np.maximum(cnt, 1), 0.0)
    A = np.vstack([mu, np.zeros((n * p, n))])
    Sig = np.diag(sig2)
    h = np.zeros(Te)
    phi = Hyper.phi0
    sigh2 = Hyper.Sh / (Hyper.nuh - 1)
    _, ymhat, Sel = missing_var(Y, A, Sig, h, p, M=M, z=z, V0=V0, m0=m0, rng=r)
    ym = ymhat
    Nm = ym.size
    if M is not None:
        M = sp.csc_array(M)
        zv = np.ravel(z)
    yo_part = Sel.So @ Sel.yo

    A_sum = np.zeros((k, n))
    Sig_sum = np.zeros((n, n))
    h_sum = np.zeros(Te)
    phi_sum = 0.0
    sigh2_sum = 0.0
    ym_sum = np.zeros(Nm)
    store_ym = np.zeros((nsim, Nm), dtype=np.float32)
    max_resid = 0.0
    n_accept = 0
    n_mh = 0
    if draws:
        D = Struct(A=np.zeros((nsim, k * n)), Sig=np.zeros((nsim, n * n)), h=np.zeros((nsim, Te)),
                   phi=np.zeros(nsim), sigh2=np.zeros(nsim))

    for isim in range(1, nsim + burnin + 1):
            # ---- BLOCK 1: the missing values ----
        if isim > 1:
            ym = missing_var(Y, A, Sig, h, p, M=M, z=z, V0=V0, m0=m0, rng=r)[0]
        ycomp = yo_part + Sel.Sm @ ym
        Yt, X = build_lags(ycomp.reshape(T, n), p)

            # ---- BLOCK 2: the coefficients, jointly ----
        w = np.exp(-h)
        A = var_coef_joint(Yt, X, Sig, w, iVb, rng=r)[0]

            # ---- BLOCK 3: Sig ----
        E = Yt - X @ A
        Shat = S0 + E.T @ (E * w[:, None])
        Sig = r.iwishrnd((Shat + Shat.T) / 2, nu0 + Te)

            # ---- BLOCK 4: h ----
        s2 = np.sum(solve_triangular(np.linalg.cholesky(Sig), E.T, lower=True)**2, axis=0)
        h, na, nb = csv_armh_block(s2, phi, sigh2, h, n, block=hblock, c_reject=c_reject,
                                   forced_accept=isim <= 20, rng=r)
        if isim > 20:
            n_accept += na
            n_mh += nb

            # ---- BLOCK 5: phi and sigh2 ----
        phi, sigh2, _ = sv0_params(h, phi, Hyper, 1, proposal='truncated', rng=r)

        if isim > burnin:
            i = isim - burnin - 1
            store_ym[i] = ym
            ym_sum += ym
            A_sum += A
            Sig_sum += Sig
            h_sum += h
            phi_sum += phi
            sigh2_sum += sigh2
            if M is not None:
                max_resid = max(max_resid, np.max(np.abs(M @ ycomp - zv)))
            if draws:
                D.A[i] = A.ravel(order='F')
                D.Sig[i] = Sig.ravel(order='F')
                D.h[i] = h
                D.phi[i] = phi
                D.sigh2[i] = sigh2

    res = Struct()
    ymean = ym_sum / nsim
    res.Y_mean = (yo_part + Sel.Sm @ ymean).reshape(T, n)
    res.Y_q = np.repeat(res.Y_mean[:, :, None], probs.size, axis=2)
    if Nm > 0:
        qv = np.quantile(store_ym.astype(float), probs, axis=0, method='hazen')
        for j in range(probs.size):
            res.Y_q[:, :, j] = (yo_part + Sel.Sm @ qv[j]).reshape(T, n)
    res.A_mean = A_sum / nsim
    res.Sig_mean = Sig_sum / nsim
    res.h_mean = h_sum / nsim
    res.phi_mean = phi_sum / nsim
    res.sigh2_mean = sigh2_sum / nsim
    res.accept_rate = n_accept / max(n_mh, 1)
    res.max_resid = max_resid
    res.sig2 = sig2
    res.quantiles = probs
    res.prior = Struct(kappa=np.asarray(kappa, dtype=float), V=V, nu0=nu0, S0=S0, Hyper=Hyper,
                       V0=V0, m0=m0)
    res.nsim = nsim
    res.burnin = burnin
    res.seed = seed
    res.p = p
    res.c_reject = c_reject
    res.hblock = hblock
    if draws:
        D.ym = store_ym
        res.draws = D
    return res


def _ar4_resid_var(y):
    """Residual variance of an AR(4) with intercept, over the periods whose value and four
    lags are all observed; NaN when fewer than 20 such periods exist."""
    T = y.size
    Z = np.column_stack([np.ones(T - 4), y[3:T - 1], y[2:T - 2], y[1:T - 3], y[0:T - 4]])
    yy = y[4:T]
    ok = np.all(~np.isnan(np.column_stack([Z, yy])), axis=1)
    if ok.sum() < 20:
        return np.nan
    b = np.linalg.lstsq(Z[ok], yy[ok], rcond=None)[0]
    return np.mean((yy[ok] - Z[ok] @ b)**2)
