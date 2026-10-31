import numpy as np


def predictive(A, Sig, ylag, H, tgt=None):
    """bvt.forecast.predictive - mean and standard deviation of the h-step predictive
    distribution of a reduced-form VAR, one row per posterior draw. Given the coefficients and
    the error covariance matrix, the h-step forecast is Gaussian.

      mu, sd = bvt.forecast.predictive(A, Sig, ylag, H)
      mu, sd = bvt.forecast.predictive(A, Sig, ylag, H, tgt)

      A     : nsim x n*k draws of the reduced-form coefficients, each row the k x n matrix
              stacked column by column with the intercept first, k = n*p + 1, as
              bvt.structural.reduced_form returns them
      Sig   : nsim x n x n draws of the error covariance matrix
      ylag  : n x p matrix of the last p observations, most recent column first
      H     : longest horizon
      tgt   : variables to return, counted from 0 (default all n)
      mu,sd : nsim x len(tgt) x H

    The predictive density of a variable is the mixture over rows, whose quantiles
    bvt.forecast.mixquantile computes.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 12.2.3.
    """
    A = np.asarray(A, dtype=float)
    Sig = np.asarray(Sig, dtype=float)
    ylag = np.asarray(ylag, dtype=float)
    if ylag.ndim == 1:
        ylag = ylag[:, None]
    nsim = A.shape[0]
    n = Sig.shape[1]
    k = A.shape[1] // n
    p = (k - 1) // n
    if k * n != A.shape[1] or p * n + 1 != k or p < 1:
        raise ValueError('A must have n*(n*p+1) columns for an integer p')
    if Sig.shape != (nsim, n, n):
        raise ValueError('Sig must be nsim x n x n')
    if ylag.shape != (n, p):
        raise ValueError(f'ylag must be {n} by {p}')
    tgt = np.arange(n) if tgt is None else np.asarray(tgt, dtype=int).ravel()
    if np.any(tgt < 0) or np.any(tgt >= n):
        raise ValueError(f'tgt must index the {n} variables')
    H = int(H)
    nt = tgt.size
    mu = np.zeros((nsim, nt, H))
    sd = np.zeros((nsim, nt, H))
    chunk = 2000                                    # draws handled at once, to bound memory
    for d0 in range(0, nsim, chunk):
        d = slice(d0, min(d0 + chunk, nsim))
        Ad = A[d].reshape(-1, n, k).transpose(0, 2, 1)          # (m, k, n)
        c = Ad[:, 0, :]
        Phi = Ad[:, 1:, :].reshape(-1, p, n, n).transpose(0, 1, 3, 2)   # Phi[:, l-1] = Phi_l
        Sd = Sig[d]
        m = Ad.shape[0]
        Psi = np.zeros((H, m, n, n))
        Psi[0] = np.eye(n)
        yl = np.tile(ylag, (m, 1, 1))
        V = np.zeros((m, n, n))
        for h in range(1, H + 1):
            yh = c.copy()
            for lag in range(1, p + 1):
                yh = yh + (Phi[:, lag - 1] @ yl[:, :, lag - 1:lag])[:, :, 0]
            yl = np.concatenate([yh[:, :, None], yl[:, :, :p - 1]], axis=2)
            for lag in range(1, min(h - 1, p) + 1):
                Psi[h - 1] = Psi[h - 1] + Phi[:, lag - 1] @ Psi[h - 1 - lag]
            V = V + Psi[h - 1] @ Sd @ Psi[h - 1].transpose(0, 2, 1)
            mu[d, :, h - 1] = yh[:, tgt]
            sd[d, :, h - 1] = np.sqrt(np.diagonal(V, axis1=1, axis2=2)[:, tgt])
    return mu, sd
