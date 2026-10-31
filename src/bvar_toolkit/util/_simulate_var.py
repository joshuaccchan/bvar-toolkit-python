import numpy as np

from .._rng import as_rng


def simulate_var(A, Sig, T, Y0=None, rng=None):
    """bvt.util.simulate_var - simulates T observations of a VAR with Gaussian errors, starting
    from its stationary distribution or from given initial values. Written for this toolkit.

      Y, Y0 = bvt.util.simulate_var(A, Sig, T)
      Y, Y0 = bvt.util.simulate_var(A, Sig, T, Y0)
      Y, Y0 = bvt.util.simulate_var(A, Sig, T, Y0, rng)

      A   : k x n coefficients, intercepts first, then the lag-1 block, ..., the lag-p block, with
            k = 1 + n*p (the layout of bvt.util.build_lags)
      Sig : n x n error covariance matrix, positive definite
      T   : number of observations to simulate
      Y0  : p x n initial values y_{1-p}, ..., y_0, oldest first. Given as an input, the path
            starts from them; otherwise they are drawn from the stationary distribution, which
            needs a stable VAR, and np.vstack([Y0, Y]) is then a sample of the stationary process
      rng : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      Y   : T x n simulated observations y_1, ..., y_T

    The draws come from rng.randn: n*p for the initial values when they are drawn, then n per
    period.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications. Chapman &
    Hall/CRC.
    """
    r = as_rng(rng)
    A = np.asarray(A, dtype=float)
    Sig = np.asarray(Sig, dtype=float)
    k, n = A.shape
    p = (k - 1) / n
    if p < 1 or p != round(p):
        raise ValueError(f'A must have 1 + n*p rows for its n = {n} columns')
    p = int(p)
    T = int(T)
    if Y0 is None:
        F = np.vstack([A[1:].T, np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])])
        if np.max(np.abs(np.linalg.eigvals(F))) >= 1:
            raise ValueError('the VAR is not stable, so it has no stationary distribution; '
                             'pass initial values Y0')
        c = np.concatenate([A[0], np.zeros(n * (p - 1))])
        mu = np.linalg.solve(np.eye(n * p) - F, c)          # mean of (y_t', ..., y_{t-p+1}')'
        V = np.zeros((n * p, n * p))                         # V = sum_j F^j Q F^j', by doubling
        V[:n, :n] = Sig
        Fj = F
        for _ in range(200):
            Vnew = V + Fj @ V @ Fj.T
            Fj = Fj @ Fj
            done = np.max(np.abs(Vnew - V)) <= 1e-15 * np.max(np.abs(Vnew))
            V = Vnew
            if done:
                break
        s = mu + np.linalg.cholesky((V + V.T) / 2) @ r.randn(n * p)  # (y_0', ..., y_{1-p}')'
        Y0 = np.flipud(s.reshape(p, n))
    else:
        Y0 = np.asarray(Y0, dtype=float)
        if Y0.shape != (p, n):
            raise ValueError(f'Y0 must be p x n = {p} x {n}')
    C = np.linalg.cholesky(Sig)
    Yfull = np.vstack([Y0, np.zeros((T, n))])
    for t in range(p, p + T):
        x = np.concatenate([[1.0], np.flipud(Yfull[t - p:t]).ravel()])  # (1, y_{t-1}', ..., y_{t-p}')
        Yfull[t] = x @ A + C @ r.randn(n)
    return Yfull[p:], Y0
