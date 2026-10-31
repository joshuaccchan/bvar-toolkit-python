from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .._linalg import BandChol
from .._rng import as_rng
from .._struct import Struct
from ..util import select_obs


def missing_var(Y, A, Sig, h, p, M=None, z=None, o=None, V0=100, m0=0, ndraws=1, rng=None):
    """bvt.samplers.missing_var - draw the missing values of a VAR(p) in one block, given the
    parameters, subject to linear restrictions M @ y = z on the stacked data, hard or soft.

      ym, ymhat, S = bvt.samplers.missing_var(Y, A, Sig, h, p)
      ym, ymhat, S = bvt.samplers.missing_var(..., M=M, z=z, V0=100, m0=0, ndraws=1, rng=None)
      ym, ymhat, S = bvt.samplers.missing_var(..., M=M, z=z, o=o)

      Y     : T x n data, NaN wherever a value is missing; the first p rows are the
              initial conditions and may have missing values too
      A     : k x n VAR coefficients, k = 1 + n*p, intercept first, in the order of
              bvt.util.build_lags, so that y_t' = x_t' @ A + e_t'
      Sig   : n x n error covariance scale, e_t ~ N(0, exp(h_t)*Sig) for t = p+1, ..., T
      h     : (T-p)-vector of log-volatilities, or None for a constant covariance
      M, z  : q x Tn restriction matrix and q-vector of values on y = vec(Y') = Y.ravel(),
              the stacking of bvt.util.select_obs; every row must involve a missing value.
              Default none
      o     : variances of the errors in soft restrictions z = M @ y + e, e ~ N(0, diag(o)),
              a scalar or a q-vector; default None, which makes the restrictions hard,
              M @ y = z
      V0, m0 : prior variance and mean of the missing values in the first p rows, a scalar
              or one entry per variable; defaults 100 and 0
      ndraws : number of independent draws, default 1
      rng   : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      ym    : Nm x ndraws draws of the missing values, in the order of select_obs; an
              Nm-vector when ndraws is 1
      ymhat : Nm-vector, the conditional mean, restrictions included
      S     : Struct with So, Sm and yo from select_obs, so that y = S.So @ S.yo + S.Sm @ ym

    Given the parameters, the missing values are Gaussian with a banded precision matrix K;
    one lower Cholesky factor of K gives the mean, the draw and the update that imposes
    M @ y = z exactly. Soft restrictions add Mm.T @ diag(o)^{-1} @ Mm to K, with Mm = M @ Sm,
    and need no update. One call consumes Nm*ndraws standard normals. Written for this toolkit.

    See:
    Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian
    State Space Models with Missing Data, Journal of Econometrics, 236(1): 105468,
    Sections 2.1-2.2 and Appendix D.
    Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications,
    Chapman & Hall/CRC, Algorithm 2.6.
    """
    Y = np.asarray(Y, dtype=float)
    A = np.asarray(A, dtype=float)
    Sig = np.atleast_2d(np.asarray(Sig, dtype=float))
    T, n = Y.shape
    k = 1 + n * p
    Te = T - p
    if Te < 1 or A.shape != (k, n) or Sig.shape != (n, n):
        raise ValueError(f'Y must have more than p rows, A must be {k} x {n} and Sig {n} x {n}')
    h = np.zeros(Te) if h is None or np.size(h) == 0 else np.asarray(h, dtype=float).ravel()
    if h.size != Te:
        raise ValueError(f'h must have T - p = {Te} entries')
    if (M is None) != (z is None) or (M is not None and (M.shape[1] != T * n
                                                          or M.shape[0] != np.size(z))):
        raise ValueError(f'M must be q x {T * n} and z a q-vector, given together')
    if o is not None:
        ov = np.asarray(o, dtype=float).ravel()
        if M is None or not (ov.size == 1 or ov.size == np.size(z)) \
                or not np.all((ov > 0) & np.isfinite(ov)):
            raise ValueError('o must come with M and z and hold one positive variance, or one '
                             'per row of M')
    V0 = np.ravel(V0) * np.ones(n)
    m0 = np.ravel(m0) * np.ones(n)

    So, Sm, yo = select_obs(Y)
    S = Struct(So=So, Sm=Sm, yo=yo)
    Nm = Sm.shape[1]
    if Nm == 0:
        return (np.zeros(0) if ndraws == 1 else np.zeros((0, ndraws))), np.zeros(0), S

    # H @ y - 1 kron b0 = e for periods p+1, ..., T, with y_t = b0 + B_1 y_{t-1} + ... + e_t:
    # block (t, t+p-lag) of H is I for lag 0 and -B_lag = -A[1+(lag-1)n:1+lag*n].T after it
    blocks = np.concatenate([np.eye(n)[None], -A[1:].reshape(p, n, n).transpose(0, 2, 1)])
    rows, cols = _pattern(Te, p, n)
    vals = np.broadcast_to(blocks, (Te, p + 1, n, n)).ravel()
    keep = vals != 0
    H = sp.csc_array((vals[keep], (rows[keep], cols[keep])), shape=(Te * n, T * n))
    iSig = np.linalg.solve(Sig, np.eye(n))
    iSig = (iSig + iSig.T) / 2
    iOm = sp.kron(sp.diags_array(np.exp(-h)), sp.csc_array(iSig), format='csc')
    im = np.flatnonzero(np.isnan(Y.ravel()))
    yfill = So @ yo                                      # y with zeros where values are missing
    Gm = H[:, im]                                        # H @ Sm
    r = np.tile(A[0], Te) - H @ yfill
    K = Gm.T @ iOm @ Gm
    c = Gm.T @ (iOm @ r)

    # the prior on the missing initial conditions
    tm, vm = im // n, im % n
    init = np.flatnonzero(tm < p)
    if init.size:
        K = K + sp.csc_array((1 / V0[vm[init]], (init, init)), shape=(Nm, Nm))
        c[init] = c[init] + m0[vm[init]] / V0[vm[init]]

    if M is not None:
        M = sp.csc_array(M)
        Mm = M @ Sm
        if np.any(np.asarray(abs(Mm).sum(axis=1)).ravel() == 0):
            raise ValueError('a row of M involves no missing value')
        zm = np.ravel(z) - M @ yfill
        if o is not None:                                # soft: z enters as observations
            iO = sp.diags_array(1 / (ov * np.ones(zm.size)))
            K = K + Mm.T @ iO @ Mm
            c = c + Mm.T @ (iO @ zm)

    C = BandChol(K)
    ymhat = C.solve_t(C.solve(c))
    u = as_rng(rng).randn(Nm, ndraws)
    ym = ymhat[:, None] + C.solve_t(u)

    if M is not None and o is None:                      # hard: the update imposes M @ y = z
        U = C.solve_t(C.solve(Mm.T.toarray()))
        MU = Mm @ U
        dz = np.linalg.solve(MU, np.column_stack([zm - Mm @ ymhat, zm[:, None] - Mm @ ym]))
        ymhat = ymhat + U @ dz[:, 0]
        ym = ym + U @ dz[:, 1:]
    return (ym[:, 0] if ndraws == 1 else ym), ymhat, S


@lru_cache(maxsize=8)
def _pattern(Te, p, n):
    """Row and column of every entry of the blocks (t, t+p-lag) of H, lag = 0, ..., p, in the
    order of np.broadcast_to(blocks, (Te, p+1, n, n)).ravel()."""
    t, lag, i, j = np.ix_(np.arange(Te), np.arange(p + 1), np.arange(n), np.arange(n))
    rows = np.broadcast_to(t * n + i, (Te, p + 1, n, n)).ravel()
    cols = np.broadcast_to((t + p - lag) * n + j, (Te, p + 1, n, n)).ravel()
    rows.flags.writeable = False
    cols.flags.writeable = False
    return rows, cols
