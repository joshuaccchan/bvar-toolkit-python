import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import spsolve_triangular
from scipy.special import gammaln

from .._rng import as_rng


def iterate(branch, draw, cfg, rng=None):
    """bvt.forecast.iterate - one iterated-forecast and predictive-likelihood step
    for ONE posterior draw, dispatched to an internal NAMED branch per
    error/volatility specification. Call it once per kept draw: each call
    simulates one path and evaluates the predictive density analytically given it.

      fc = bvt.forecast.iterate(branch, draw, cfg)
      fc = bvt.forecast.iterate(branch, draw, cfg, rng)

    branch - which specification to run (see BRANCHES below)
    draw   - per-draw posterior quantities, a dict; the keys read are listed per branch
    cfg    - vintage-level constants, fixed across draws, a dict
    rng    - None, an integer seed, a numpy.random.Generator or a bvt.Stream
    fc     - 2 x (2n+1), one row per evaluated horizon: [EYtp1 (n point
             forecast), lden (n per-variable log predictive likelihoods),
             lden_joint (joint log predictive likelihood over ALL n modeled
             variables)]. 'mahp_sv' gives h=1 then h=4; 'springer_*' gives h=0
             (nowcast) then h=1. A row whose evaluation guard is off this vintage
             (see Horizons) comes back as zeros, and its simulation draws
             are still consumed. Subsetting is the CALLER's job: pass realized
             values already subsetted, or select columns in bvt.forecast.tables.

    Horizons: each branch runs its step loop in full (mahp_sv: tt = 1:4, evaluated
    at tt = 1 and 4; springer_*: tt = 1:2, evaluated at h=0 and h=1). The first
    evaluated step is unguarded; each later step tt is evaluated only when t <= T
    - tt, which in the springer branches skips the h=1 evaluation at t = T-1 even
    though the realized value exists. Scheme and quirk are preserved deliberately.
    The periods t and T count from 1.

    Missing latest observation: when cfg['is_last_miss'] is true the branch advances
    the state one extra simulation step before the tt loop, so that h=0 evaluates
    against the quarter after the last OBSERVED one; that step consumes rng draws.

    BRANCHES: what each one models, and the draw/cfg keys it reads.

    'mahp_sv'  Structural BVAR with per-variable random-walk SV.
       draw: alp (k_alp-vector), beta (k_beta-vector), h_T (n-vector), Sigh (n-vector)
       cfg:  Yt (estimation sample; the last p rows feed the lag stack), Y (full
             realized value matrix; rows t and t+3, counted from 0, are read), p, t, T

    'springer_gauss'  Homoskedastic Gaussian errors.
       draw: A (k x n coefficient matrix; a caller holding beta = vec(A) passes
             beta.reshape((k, n), order='F')); CSig, the lower Cholesky factor of Sig,
             dense or sparse (sp.diags_array(np.sqrt(Sig_hat)) when Sig is diagonal);
             dSig (n), the diagonal of Sig
       cfg:  shortYt, data_tpk (>= 2 rows; pre-subsetted when the model covers a
             subset of the columns of realized values), is_last_miss, p, t, T

    'springer_csv'  Gaussian errors with common stochastic volatility (CSV).
       draw: A, CSig (the lower Cholesky factor of Sig), Sig, h (Tt-vector path; only
             h[-1] is read), rho, sigh2
       cfg:  shortYt, data_tpk, is_last_miss, p, t, T

    'springer_csv_t'  CSV with Student-t errors.
       draw: A, CSig, Sig, h, rho, sigh2, nu
       cfg:  shortYt, data_tpk, is_last_miss, p, t, T

    'springer_csv_t_ma'  CSV with Student-t and MA(1) errors.
       draw: A, CSig, Sig, h, rho, sigh2, nu, psi, Hpsi (the sparse Tt x Tt MA
             rotation held by the sampler)
       cfg:  shortYt, Z (the Tt x k estimation design), data_tpk, is_last_miss,
             p, t, T

    See:
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for
    Large Bayesian VARs, International Journal of Forecasting, 37(3), 1212-1226.
    Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Ed.),
    Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham.
    """
    branches = {'mahp_sv': _mahp_sv, 'springer_gauss': _springer_gauss,
                'springer_csv': _springer_csv, 'springer_csv_t': _springer_csv_t,
                'springer_csv_t_ma': _springer_csv_t_ma}
    if branch not in branches:
        raise ValueError(f"unknown branch '{branch}'; use mahp_sv, springer_gauss, "
                         'springer_csv, springer_csv_t or springer_csv_t_ma')
    return branches[branch](draw, cfg, as_rng(rng))


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)


def _lags(Y, p):
    """[1, y_T, y_{T-1}, ..., y_{T-p+1}] from the last p rows of Y."""
    return np.r_[1.0, np.asarray(Y, dtype=float)[::-1][:p].ravel()]


def _gauss_rows(CS, yobs, EY, dS, ldet):
    tmpu = solve_triangular(CS, yobs - EY, lower=True)
    n = EY.size
    lden_joint = -n / 2 * np.log(2 * np.pi) - ldet - .5 * (tmpu @ tmpu)
    lden = -.5 * np.log(2 * np.pi * dS) - .5 * (yobs - EY)**2 / dS
    return np.r_[EY, lden, lden_joint]


def _mahp_sv(draw, cfg, r):
    alp = np.asarray(draw['alp'], dtype=float).ravel()
    beta = np.asarray(draw['beta'], dtype=float).ravel()
    h_Tp1 = np.asarray(draw['h_T'], dtype=float).ravel()
    Sigh = np.asarray(draw['Sigh'], dtype=float).ravel()
    Yt, Y = np.asarray(cfg['Yt'], dtype=float), np.asarray(cfg['Y'], dtype=float)
    p, t, T = cfg['p'], cfg['t'], cfg['T']
    n = Y.shape[1]
    A = np.eye(n)
    A[np.tril_indices(n, -1)] = alp                   # the strict lower triangle, row by row
    fc = np.zeros((2, 2 * n + 1))
        # transform the parameters into reduced form
    sqrtSigh = np.sqrt(Sigh)

    def cov(h):
        AD = solve_triangular(A, np.diag(np.exp(h)), lower=True)
        return solve_triangular(A, AD.T, lower=True).T

    h_Tp1 = h_Tp1 + sqrtSigh * r.randn(n)
    S = cov(h_Tp1)
    B = solve_triangular(A, beta.reshape((n * p + 1, n), order='F').T, lower=True).T
    xtp1 = _lags(Yt, p)
    CS = np.linalg.cholesky(S)
    for tt in range(1, 5):
        EYtp1 = xtp1 @ B
        dS = np.diag(S)
        if tt == 1:
            fc[0] = _gauss_rows(CS, Y[t], EYtp1, dS, np.sum(np.log(np.diag(CS))))
        elif tt == 4 and t <= T - tt:
            fc[1] = _gauss_rows(CS, Y[t + 3], EYtp1, dS, np.sum(np.log(np.diag(CS))))
        Ytp1 = EYtp1 + CS @ r.randn(n)
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]

        h_Tp1 = h_Tp1 + sqrtSigh * r.randn(n)
        S = cov(h_Tp1)
        CS = np.linalg.cholesky(S)
    return fc


def _springer_gauss(draw, cfg, r):
    A = np.asarray(draw['A'], dtype=float)
    CSig = _dense(draw['CSig'])
    dSig = np.asarray(draw['dSig'], dtype=float).ravel()
    data_tpk = np.asarray(cfg['data_tpk'], dtype=float)
    p, t, T = cfg['p'], cfg['t'], cfg['T']
    n = A.shape[1]
    fc = np.zeros((2, 2 * n + 1))
    ldet = np.sum(np.log(np.diag(CSig)))
    xtp1 = _lags(cfg['shortYt'], p)
    if cfg['is_last_miss']:      # if the lastest data are missing, do one more forecast horizon
        Ytp1 = xtp1 @ A + CSig @ r.randn(n)
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    for tt in range(1, 3):
        EYtp1 = xtp1 @ A
        if tt == 1:
            fc[0] = _gauss_rows(CSig, data_tpk[0], EYtp1, dSig, ldet)
        elif tt == 2 and t <= T - tt:
            fc[1] = _gauss_rows(CSig, data_tpk[1], EYtp1, dSig, ldet)
        Ytp1 = EYtp1 + CSig @ r.randn(n)
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    return fc


def _csv_rows(CSig, yobs, EY, dSig, htp1, ldet):
    tmpu = solve_triangular(CSig, yobs - EY, lower=True)
    n = EY.size
    lden_joint = -n / 2 * np.log(2 * np.pi) - n / 2 * htp1 - ldet \
        - .5 * (tmpu @ tmpu) / np.exp(htp1)
    lden = -.5 * np.log(2 * np.pi * dSig) - .5 * (yobs - EY)**2 / dSig
    return np.r_[EY, lden, lden_joint]


def _springer_csv(draw, cfg, r):
    A = np.asarray(draw['A'], dtype=float)
    CSig = _dense(draw['CSig'])
    Sig = np.asarray(draw['Sig'], dtype=float)
    rho, sigh2 = draw['rho'], draw['sigh2']
    data_tpk = np.asarray(cfg['data_tpk'], dtype=float)
    p, t, T = cfg['p'], cfg['t'], cfg['T']
    n = A.shape[1]
    fc = np.zeros((2, 2 * n + 1))
    ldet = np.sum(np.log(np.diag(CSig)))
    xtp1 = _lags(cfg['shortYt'], p)
    htp1 = np.asarray(draw['h'], dtype=float).ravel()[-1]
    if cfg['is_last_miss']:      # if the lastest data are missing, do one more forecast horizon
        htp1 = rho * htp1 + np.sqrt(sigh2) * r.randn()
        EYtp1 = xtp1 @ A
        Ytp1 = EYtp1 + (np.exp(htp1 / 2) * CSig) @ r.randn(n)
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    for tt in range(1, 3):
        htp1 = rho * htp1 + np.sqrt(sigh2) * r.randn()
        EYtp1 = xtp1 @ A
        dSig = np.exp(htp1) * np.diag(Sig)
        if tt == 1:
            fc[0] = _csv_rows(CSig, data_tpk[0], EYtp1, dSig, htp1, ldet)
        elif tt == 2 and t <= T - tt:
            fc[1] = _csv_rows(CSig, data_tpk[1], EYtp1, dSig, htp1, ldet)
        Ytp1 = EYtp1 + (np.exp(htp1 / 2) * CSig) @ r.randn(n)
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    return fc


def _t_rows(CSig, yobs, EY, dSig, htp1, ldet, nu):
    n = EY.size
    ct = gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi * dSig)
    ct_joint = gammaln((nu + n) / 2) - gammaln(nu / 2) - n / 2 * np.log(nu * np.pi)
    tmpu = solve_triangular(CSig, yobs - EY, lower=True)
    lden_joint = ct_joint - ldet - n / 2 * htp1 \
        - (nu + n) / 2 * np.log(1 + (tmpu @ tmpu) / nu / np.exp(htp1))
    lden = ct - (nu + 1) / 2 * np.log(1 + (yobs - EY)**2 / dSig / nu)
    return np.r_[EY, lden, lden_joint]


def _springer_csv_t(draw, cfg, r):
    A = np.asarray(draw['A'], dtype=float)
    CSig = _dense(draw['CSig'])
    Sig = np.asarray(draw['Sig'], dtype=float)
    rho, sigh2, nu = draw['rho'], draw['sigh2'], draw['nu']
    data_tpk = np.asarray(cfg['data_tpk'], dtype=float)
    p, t, T = cfg['p'], cfg['t'], cfg['T']
    n = A.shape[1]
    fc = np.zeros((2, 2 * n + 1))
    ldet = np.sum(np.log(np.diag(CSig)))
    xtp1 = _lags(cfg['shortYt'], p)
    htp1 = np.asarray(draw['h'], dtype=float).ravel()[-1]
    if cfg['is_last_miss']:      # if the lastest data are missing, do one more forecast horizon
        htp1 = rho * htp1 + np.sqrt(sigh2) * r.randn()
        EYtp1 = xtp1 @ A
        z = (np.exp(htp1 / 2) * CSig) @ r.randn(n)
        Ytp1 = EYtp1 + z / np.sqrt(r.gamrnd(nu / 2, 2 / nu))
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    for tt in range(1, 3):
        htp1 = rho * htp1 + np.sqrt(sigh2) * r.randn()
        EYtp1 = xtp1 @ A
        dSig = np.exp(htp1) * np.diag(Sig)
        if tt == 1:
            fc[0] = _t_rows(CSig, data_tpk[0], EYtp1, dSig, htp1, ldet, nu)
        elif tt == 2 and t <= T - tt:
            fc[1] = _t_rows(CSig, data_tpk[1], EYtp1, dSig, htp1, ldet, nu)
        z = (np.exp(htp1 / 2) * CSig) @ r.randn(n)
        Ytp1 = EYtp1 + z / np.sqrt(r.gamrnd(nu / 2, 2 / nu))
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    return fc


def _springer_csv_t_ma(draw, cfg, r):
    A = np.asarray(draw['A'], dtype=float)
    CSig = _dense(draw['CSig'])
    Sig = np.asarray(draw['Sig'], dtype=float)
    rho, sigh2, nu, psi = draw['rho'], draw['sigh2'], draw['nu'], draw['psi']
    shortYt = np.asarray(cfg['shortYt'], dtype=float)
    data_tpk = np.asarray(cfg['data_tpk'], dtype=float)
    p, t, T = cfg['p'], cfg['t'], cfg['T']
    n = A.shape[1]
    fc = np.zeros((2, 2 * n + 1))
    ldet = np.sum(np.log(np.diag(CSig)))
    xtp1 = _lags(shortYt, p)
    E = spsolve_triangular(sp.csr_array(draw['Hpsi']), shortYt - np.asarray(cfg['Z']) @ A,
                           lower=True)
    etp1 = E[-1]
    htp1 = np.asarray(draw['h'], dtype=float).ravel()[-1]
    if cfg['is_last_miss']:      # if the lastest data are missing, do one more forecast horizon
        htp1 = rho * htp1 + np.sqrt(sigh2) * r.randn()
        EYtp1 = xtp1 @ A + psi * etp1
        z = (np.exp(htp1 / 2) * CSig) @ r.randn(n)
        etp1 = z / np.sqrt(r.gamrnd(nu / 2, 2 / nu))
        Ytp1 = EYtp1 + etp1
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    for tt in range(1, 3):
        htp1 = rho * htp1 + np.sqrt(sigh2) * r.randn()
        EYtp1 = xtp1 @ A + psi * etp1
        dSig = np.exp(htp1) * np.diag(Sig)
        if tt == 1:
            fc[0] = _t_rows(CSig, data_tpk[0], EYtp1, dSig, htp1, ldet, nu)
        elif tt == 2 and t <= T - tt:
            fc[1] = _t_rows(CSig, data_tpk[1], EYtp1, dSig, htp1, ldet, nu)
        z = (np.exp(htp1 / 2) * CSig) @ r.randn(n)
        etp1 = z / np.sqrt(r.gamrnd(nu / 2, 2 / nu))
        Ytp1 = EYtp1 + etp1
        xtp1 = np.r_[1.0, Ytp1, xtp1[1:-n]]
    return fc
