import numpy as np


def check_separable(S, Rineq=None):
    """bvt.structural.check_separable - test whether a set of sign and ranking restrictions
    separates every pair of shocks, as Assumption 2 of Chan, Matthes and Yu (2026) requires
    of the restrictions given to sign_assign.

      ok, bad = bvt.structural.check_separable(S)
      ok, bad = bvt.structural.check_separable(S, Rineq)

      S     : n x m sign restrictions, as bvt.structural.sign_assign takes them
      Rineq : m x n x k ranking restrictions, as sign_assign takes them; default none
      ok    : True when every pair of shocks is separated
      bad   : the pairs that are not, one row [j, l] per pair, j < l, counted from 0

    Two shocks are separated when some variable is restricted with the same sign under both
    and another with opposite signs, or when the same pair of variables is ranked one way for
    one shock and the other way for the other. When a pair is not separated, sign_assign can
    raise an error. Call this once before the rejection loop.

    Called with S alone it reports the pairs the ranking restrictions are carrying.

    See:
    Chan, J.C.C., Matthes, C. and Yu, X. (2026). Large Structural VARs with Multiple Sign and
    Ranking Restrictions, Quantitative Economics, 17(3): 709-740, Assumption 2.
    """
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    if Rineq is None or np.size(Rineq) == 0:
        Rineq = np.zeros((m, n))
    Rineq = np.asarray(Rineq, dtype=float)
    if Rineq.ndim == 2:
        Rineq = Rineq[:, :, None]
    if Rineq.shape[:2] != (m, n):
        raise ValueError(f'Rineq must be m x n x k with m = {m} shocks and n = {n} variables')
    k = Rineq.shape[2]
    bad = []
    for a in range(m - 1):
        for b in range(a + 1, m):
            both = ~np.isnan(S[:, a]) & ~np.isnan(S[:, b])
            by_sign = np.any(both & (S[:, a] == S[:, b])) and np.any(both & (S[:, a] == -S[:, b]))
            by_rank = False
            for ja in range(k):
                ra = Rineq[a, :, ja]
                if not np.any(ra):
                    continue
                for jb in range(k):
                    if np.array_equal(np.sign(ra), -np.sign(Rineq[b, :, jb])):
                        by_rank = True
                        break
                if by_rank:
                    break
            if not (by_sign or by_rank):
                bad.append([a, b])
    return len(bad) == 0, np.array(bad, dtype=int).reshape(-1, 2)
