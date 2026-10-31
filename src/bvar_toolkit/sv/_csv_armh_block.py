import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded
from scipy.linalg.lapack import dtbtrs

from .._rng import as_rng


def csv_armh_block(s2, rho, sigh2, h, n, block=60, c_reject=3, forced_accept=False,
                   max_iter_mode=500, max_iter_ar=1000, rng=None):
    """bvt.sv.csv_armh_block - one sweep of accept-reject Metropolis-Hastings updates of the
    common stochastic volatility path h, one block of consecutive periods at a time.

      h, n_accept, n_block = bvt.sv.csv_armh_block(s2, rho, sigh2, h, n)
      h, n_accept, n_block = bvt.sv.csv_armh_block(..., block=60, c_reject=3, rng=None)

      s2          : T-vector, e_t' @ Sig^{-1} @ e_t at each t
      rho, sigh2  : AR(1) coefficient and innovation variance of the zero-mean h, whose
                    first value has the stationary distribution
      h, n        : current log-volatility path (T-vector) and number of series
      block       : block length, default 60; unless block >= T, the first block has a
                    random length between 1 and block, drawn with rng.randi
      c_reject    : c in the envelope target <= c * proposal, default 3
      forced_accept : take every proposal regardless of the MH ratio, default False
      max_iter_mode, max_iter_ar : caps on the two loops, defaults 500 and 1000
      rng         : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      n_accept, n_block : blocks whose proposal was accepted, and blocks in the sweep

    Each block is drawn from its conditional given the rest of the path, with the proposal
    of bvt.sv.csv_armh, a Gaussian at the mode of that conditional. With block >= T the
    path is one block, and the update differs from bvt.sv.csv_armh only in evaluating the
    proposal precision at the mode. Written for this toolkit.

    See:
    Chan, J.C.C. (2017). The Stochastic Volatility in Mean Model with Time-Varying
    Parameters: An Application to Inflation Modeling, Journal of Business and Economic
    Statistics, 35(1): 17-28.
    Chan, J.C.C. (2020). Large Bayesian VARs: A Flexible Kronecker Error Covariance
    Structure, Journal of Business and Economic Statistics, 38(1): 68-79.
    """
    L = block
    tol_mode = 1e-3
    if not (np.ndim(L) == 0 and L >= 1 and L == int(L)):
        raise ValueError('block must be a positive integer')
    if not (np.ndim(c_reject) == 0 and np.isrealobj(c_reject) and np.isfinite(c_reject)
            and c_reject > 0):
        raise ValueError('c_reject must be a finite positive scalar')
    L = int(L)
    r = as_rng(rng)
    s2 = np.asarray(s2, dtype=float).ravel()
    h = np.array(h, dtype=float).ravel()
    T = s2.size
    # the precision of h, tridiagonal: Hrho.T @ diag(dv) @ Hrho with Hrho = I - rho*(lag)
    dv = np.r_[(1 - rho**2) / sigh2, np.ones(T - 1) / sigh2]
    Qd = dv.copy()
    Qd[:-1] = dv[:-1] + (-rho * dv[1:]) * (-rho)
    Qe = -rho * dv[1:]                                 # Q[t+1, t] = Q[t, t+1]

    if L >= T:
        edges = np.array([0, T])
    else:
        first = r.randi(L)
        edges = np.unique(np.r_[0, np.arange(first, T + 1, L), T])
    n_block = edges.size - 1
    n_accept = 0
    for ib in range(n_block):
        a, e = edges[ib], edges[ib + 1]               # the block is h[a:e]
        m = e - a
        d, off = Qd[a:e], Qe[a:e - 1]
        b = np.zeros(m)                                # the terms linking the block to the rest
        if a > 0:
            b[0] += Qe[a - 1] * h[a - 1]
        if e < T:
            b[-1] += Qe[e - 1] * h[e]
        sB = s2[a:e]
        x = h[a:e].copy()
        err = np.inf
        it = 0
        while not err <= tol_mode:
            it += 1
            if it > max_iter_mode:
                raise RuntimeError(f'mode search did not converge in {max_iter_mode} iterations '
                                   f'(last max|dh| = {err:g})')
            G = .5 * sB * np.exp(-x)
            xnew = solveh_banded(_band(d + G, off), G * x - n / 2 + G - b, lower=True)
            err = np.max(np.abs(xnew - x))
            x = xnew
        G = .5 * sB * np.exp(-x)
        CK = cholesky_banded(_band(d + G, off), lower=True)
        logc = _logpi(x, d, off, b, n, sB) + np.log(c_reject)

        # accept-reject step
        it = 0
        while True:
            it += 1
            if it > max_iter_ar:
                raise RuntimeError(f'no accept-reject proposal accepted in {max_iter_ar} draws '
                                   f'at c_reject = {c_reject:g}')
            u = r.randn(m)
            xc = x + dtbtrs(CK, u[:, None], uplo='L', trans='T')[0][:, 0]
            alpARc = _logpi(xc, d, off, b, n, sB) + .5 * _quad(xc - x, d, off, G) - logc
            if alpARc > np.log(r.rand()):
                break
        # Metropolis-Hastings step
        hB = h[a:e]
        alpAR = _logpi(hB, d, off, b, n, sB) + .5 * _quad(hB - x, d, off, G) - logc
        if alpAR < 0:
            alpMH = 0.0
        elif alpARc < 0:
            alpMH = -alpAR
        else:
            alpMH = alpARc - alpAR
        if alpMH > np.log(r.rand()) or forced_accept:
            h[a:e] = xc
            n_accept += 1
    return h, n_accept, n_block


def _band(d, off):
    """Lower band storage of the symmetric tridiagonal matrix with diagonal d and
    off-diagonal off; a single row when the matrix is 1 x 1."""
    return np.vstack([d, np.r_[off, 0.0]]) if d.size > 1 else d[None, :]


def _tri(d, off, x):
    """Q @ x for the symmetric tridiagonal Q with diagonal d and off-diagonal off."""
    y = d * x
    y[:-1] += off * x[1:]
    y[1:] += off * x[:-1]
    return y


def _logpi(x, d, off, b, n, sB):
    """The log conditional density of a block, up to a constant."""
    return -.5 * x @ _tri(d, off, x) - x @ b - n / 2 * np.sum(x) - .5 * np.exp(-x) @ sB


def _quad(v, d, off, G):
    """v' @ K @ v with K = Q + diag(G)."""
    return v @ (_tri(d, off, v) + G * v)
