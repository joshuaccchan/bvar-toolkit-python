import numpy as np
import scipy.sparse as sp


def dlog_gaps(Lv):
    """bvt.util.dlog_gaps - growth rates of series in levels with gaps, and the linear
    restrictions that tie the growth rates across each gap to the change over it.

      G, M, z = bvt.util.dlog_gaps(Lv)

      Lv : T x n levels, NaN where a level is missing
      G  : (T-1) x n growth rates, G[t-1] = 100*(log Lv[t] - log Lv[t-1]), NaN wherever either
           level is missing
      M  : q x (T-1)n sparse csc_array, one row per gap with an observed level on each side,
           acting on g = vec(G') = G.ravel(), the stacking of bvt.util.select_obs
      z  : q-vector, 100 times the log change over each such gap

    A gap of m missing levels leaves m+1 growth rates missing whose sum is known; a gap at
    either end of the sample leaves no restriction. Written for this toolkit.
    """
    Lv = np.asarray(Lv, dtype=float)
    if Lv.ndim != 2 or Lv.shape[0] < 2:
        raise ValueError('Lv must be a numeric matrix with at least two rows')
    if np.any(Lv <= 0):
        raise ValueError('the levels must be positive')
    T, n = Lv.shape
    G = 100 * np.diff(np.log(Lv), axis=0)
    rows, cols, z = [], [], []
    q = 0
    for i in range(n):
        obs = np.flatnonzero(~np.isnan(Lv[:, i]))
        for j in np.flatnonzero(np.diff(obs) > 1):
            a, b = obs[j], obs[j + 1]               # observed levels around the gap
            s = np.arange(a, b)                     # the growth rates G[a:b, i]
            rows += [q] * s.size
            cols += list(s * n + i)
            z.append(100 * (np.log(Lv[b, i]) - np.log(Lv[a, i])))
            q += 1
    M = sp.csc_array((np.ones(len(rows)), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                     shape=(q, (T - 1) * n))
    return G, M, np.array(z, dtype=float)
