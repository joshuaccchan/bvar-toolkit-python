import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng
from .._struct import Struct
from ..priors import niw
from ..sv import csv_armh, sv0_params
from ..util import build_lags


def var_csv(Y0, Y, p, kappa=(.2**2, 100), nsim=10000, burnin=1000, rng=None, c_reject=3,
            draws=False):
    """bvt.models.var_csv - posterior sampler for a VAR with common stochastic
    volatility, y_t' = x_t' A + u_t' with u_t ~ N(0, exp(h_t)*Sig), h_t a zero-mean
    AR(1) and a natural conjugate prior on (A,Sig). The specification is that of
    Carriero, Clark and Marcellino (2016) as implemented in Chan (2023).

      res = bvt.models.var_csv(Y0, Y, p)
      res = bvt.models.var_csv(Y0, Y, p, nsim=2000, burnin=500, rng=1)

      Y0 : initial conditions, at least max(p,4) rows, same columns as Y
      Y  : T x n estimation sample, n >= 2, each series stationary, no NaN or Inf
      p  : lag length

      kappa  : (shrinkage, intercept variance), default (.2^2, 100)
      nsim   : draws kept, default 10000;  burnin : draws discarded first,
               default 1000
      rng    : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      c_reject : envelope constant of the accept-reject step in bvt.sv.csv_armh,
               default 3; it changes the cost of the h draw and leaves the
               target unchanged
      draws  : True also returns res.draws with one row per draw: A (nsim x k*n,
               each row A.ravel(order='F')), Sig (nsim x n^2, each row
               Sig.ravel(order='F')), h (nsim x T), phi and sigh2; default False

      res.A_mean     : k x n posterior mean of the VAR coefficients, intercept
                       first, k = 1 + n*p
      res.Sig_mean   : n x n posterior mean of the scale matrix Sig
      res.h_mean     : T-vector, the posterior mean of the log-volatility
      res.phi_mean, res.sigh2_mean : posterior means of the AR(1) parameters
      res.accept_rate : share of sweeps whose Metropolis step accepted, past the
                       first 20, which take the proposal regardless
      plus the settings: nsim, burnin, seed, p, c_reject

    One factor multiplies the whole covariance matrix, so the order of the columns
    of Y does not affect the volatility model.

    See:
    Carriero, A., Clark, T.E. and Marcellino, M. (2016). Common Drifting Volatility
    in Large Bayesian VARs, Journal of Business and Economic Statistics, 34(3):
    375-390.
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large
    Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
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
    nsim, burnin, p = int(nsim), int(burnin), int(p)
    seed = int(rng) if isinstance(rng, (int, np.integer)) else None
    r = as_rng(rng)

    k = 1 + n * p
    _, X = build_lags(np.vstack([Y0[-p:], Y]), p)

    Hyper = Struct(nuh=3, Sh=.2, phi0=.98, Vphi=.05**2)
    A0, VA0, nu0, S0, _ = niw(p, kappa, Y0, Y, 'mlvarsv_ncp')
    iVA0 = 1 / VA0
    VA0iA0 = A0 / VA0[:, None]
    A0iVA0A0 = A0.T @ (iVA0[:, None] * A0)

    phi = Hyper.phi0
    sigh2 = 1 / r.gamrnd(Hyper.nuh, 1 / Hyper.Sh)
    h = np.zeros(T)

    A_sum = np.zeros((k, n))
    Sig_sum = np.zeros((n, n))
    h_sum = np.zeros(T)
    phi_sum = 0.0
    sigh2_sum = 0.0
    n_accept = 0
    n_mh = 0
    if draws:
        D = Struct(A=np.zeros((nsim, k * n)), Sig=np.zeros((nsim, n * n)), h=np.zeros((nsim, T)),
                   phi=np.zeros(nsim), sigh2=np.zeros(nsim))

    for isim in range(1, nsim + burnin + 1):
            # ---- BLOCK 1: (A, Sig) given h, one joint draw ----
        XiOh = X.T * np.exp(-h)
        KA = XiOh @ X
        KA[np.diag_indices(k)] += iVA0
        CKA = np.linalg.cholesky(KA)
        Ahat = solve_triangular(CKA, solve_triangular(CKA, VA0iA0 + XiOh @ Y, lower=True),
                                lower=True, trans='T')
        Shat = S0 + A0iVA0A0 + (Y.T * np.exp(-h)) @ Y - Ahat.T @ KA @ Ahat
        Shat = (Shat + Shat.T) / 2                     # symmetrize against rounding
        Sig = r.iwishrnd(Shat, nu0 + T)
        CSig = np.linalg.cholesky(Sig)
        A = Ahat + solve_triangular(CKA, r.randn(k, n), lower=True, trans='T') @ CSig.T
        U = Y - X @ A
        s2 = np.sum(solve_triangular(CSig, U.T, lower=True)**2, axis=0)

            # ---- BLOCK 2: h given the VAR parameters ----
        if isim <= 20:
            h = csv_armh(s2, phi, sigh2, h, n, True, c_reject=c_reject, rng=r)[0]
        else:
            h, is_accept = csv_armh(s2, phi, sigh2, h, n, False, c_reject=c_reject, rng=r)
            n_accept += is_accept
            n_mh += 1

            # ---- BLOCK 3: the AR(1) parameters given h ----
        phi, sigh2, _ = sv0_params(h, phi, Hyper, rng=r)

        if isim > burnin:
            i = isim - burnin - 1
            A_sum += A
            Sig_sum += Sig
            h_sum += h
            phi_sum += phi
            sigh2_sum += sigh2
            if draws:
                D.A[i] = A.ravel(order='F')
                D.Sig[i] = Sig.ravel(order='F')
                D.h[i] = h
                D.phi[i] = phi
                D.sigh2[i] = sigh2

    res = Struct(A_mean=A_sum / nsim, Sig_mean=Sig_sum / nsim, h_mean=h_sum / nsim,
                 phi_mean=phi_sum / nsim, sigh2_mean=sigh2_sum / nsim,
                 accept_rate=n_accept / max(n_mh, 1), nsim=nsim, burnin=burnin, seed=seed, p=p,
                 c_reject=c_reject)
    if draws:
        res.draws = D
    return res
