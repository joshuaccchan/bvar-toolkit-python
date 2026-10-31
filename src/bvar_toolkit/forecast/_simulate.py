import numpy as np

from .._rng import as_rng


def simulate(spec, draw, cfg, rng=None):
    """bvt.forecast.simulate - h-step predictive distribution of a reduced-form VAR
    with time-varying error covariance, from ONE posterior draw, with the log
    predictive likelihood of the realized value at each horizon.

      yhat, lden, ljoint, sdev = bvt.forecast.simulate(spec, draw, cfg)
      yhat, lden, ljoint, sdev = bvt.forecast.simulate(spec, draw, cfg, rng)

      spec : which covariance the errors have
               'gauss' : constant, draw['Sig']
               'csv'   : exp(h_t)*Sig with h_t a zero-mean AR(1), the model of
                         bvt.models.var_csv
               'oisv'  : B0^{-1} diag(exp(h_t)) B0^{-T} with one zero-mean AR(1) per
                         equation, the order-invariant model of bvt.models.var_sv
      draw : one posterior draw, a dict or Struct. Every spec reads
               A    : k x n coefficients, intercept first, k = n*p + 1
             and its own volatility fields
               'gauss' : Sig (n x n)
               'csv'   : Sig (n x n), h_T, phi, sigh2 (scalars)
               'oisv'  : impact (n x n, B0), h_T, phi, sig2 (n-vectors)
      cfg  : a dict or Struct with ylag (n x p, most recent column first), H, and yobs
             (H x n, the realized values) when the scores are wanted; rows of yobs that
             are missing or not finite are skipped
      rng  : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      yhat   : H x n conditional mean, which does not depend on the volatility
      lden   : H x n log predictive likelihood of each variable given the
               simulated volatility path, NaN where yobs has no row
      ljoint : H-vector, the same jointly over all n variables
      sdev   : H x n predictive standard deviation given that path

    ONLY THE VOLATILITY IS SIMULATED. The data path is integrated out
    analytically, so lden and ljoint are exact given the simulated volatility
    path. Averaging exp(lden) over draws integrates over the parameters and the
    volatility together. With 'gauss' the result equals that of
    bvt.forecast.predictive for the same draw.

    See:
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large
    Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    spec = str(spec).lower()
    A = np.asarray(draw['A'], dtype=float)
    k, n = A.shape
    p = (k - 1) // n
    if p < 1 or p * n + 1 != k:
        raise ValueError('draw.A must be (n*p+1) by n')
    ylag = np.asarray(cfg['ylag'], dtype=float).reshape(n, -1)
    if ylag.shape != (n, p):
        raise ValueError(f'cfg.ylag must be {n} by {p}')
    H = int(cfg['H'])
    yobs = cfg.get('yobs') if isinstance(cfg, dict) else None
    if yobs is not None and np.size(yobs) == 0:
        yobs = None
    if yobs is not None:
        yobs = np.asarray(yobs, dtype=float).reshape(-1, n)

        # ---- the covariance at each horizon, the only simulated part ----
    if spec == 'gauss':
        Sg = [np.asarray(draw['Sig'], dtype=float)] * H
    elif spec == 'csv':
        r = as_rng(rng)
        h, phi, sdh = float(draw['h_T']), float(draw['phi']), np.sqrt(float(draw['sigh2']))
        Sig = np.asarray(draw['Sig'], dtype=float)
        Sg = []
        for _ in range(H):
            h = phi * h + sdh * r.randn()
            Sg.append(np.exp(h) * Sig)
    elif spec == 'oisv':
        r = as_rng(rng)
        Bi = np.linalg.inv(np.asarray(draw['impact'], dtype=float))
        h = np.asarray(draw['h_T'], dtype=float).ravel()
        phi = np.asarray(draw['phi'], dtype=float).ravel()
        sdh = np.sqrt(np.asarray(draw['sig2'], dtype=float).ravel())
        Sg = []
        for _ in range(H):
            h = phi * h + sdh * r.randn(n)
            Sg.append((Bi * np.exp(h)) @ Bi.T)
    else:
        raise ValueError(f"spec must be 'gauss', 'csv' or 'oisv'; got '{spec}'")

        # ---- the moving-average matrices, Psi[i] is Psi_i ----
    cst = A[0]
    Phi = [A[1 + lag * n:1 + (lag + 1) * n].T for lag in range(p)]
    Psi = [np.eye(n)]
    for j in range(1, H):
        Psi.append(sum(Phi[lag - 1] @ Psi[j - lag] for lag in range(1, min(j, p) + 1)))

    yhat = np.zeros((H, n))
    lden = np.full((H, n), np.nan)
    ljoint = np.full(H, np.nan)
    sdev = np.zeros((H, n))
    yl = ylag.copy()
    for j in range(H):
            # the mean iterates the VAR and does not involve the volatility
        yh = cst + sum(Phi[lag] @ yl[:, lag] for lag in range(p))
        yl = np.column_stack([yh, yl[:, :p - 1]])
        yhat[j] = yh

            # the variance given the simulated path
        V = sum(Psi[i] @ Sg[j - i] @ Psi[i].T for i in range(j + 1))
        V = (V + V.T) / 2
        dV = np.diag(V)
        sdev[j] = np.sqrt(dV)

        if yobs is not None and yobs.shape[0] > j and np.all(np.isfinite(yobs[j])):
            u = yobs[j] - yh
            lden[j] = -.5 * np.log(2 * np.pi * dV) - .5 * u**2 / dV
            CV = np.linalg.cholesky(V)
            z = np.linalg.solve(CV, u)
            ljoint[j] = -n / 2 * np.log(2 * np.pi) - np.sum(np.log(np.diag(CV))) - .5 * (z @ z)
    return yhat, lden, ljoint, sdev
