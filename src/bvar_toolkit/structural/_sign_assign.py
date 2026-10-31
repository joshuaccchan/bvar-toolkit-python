import numpy as np

from .._rng import as_rng


def sign_assign(L, S, Rineq, k=1, rng=None):
    """bvt.structural.sign_assign - accept a candidate impact matrix when every shock has at
    least one column satisfying its restrictions, then draw one such assignment at random.
    Sign and ranking restrictions are tested together.

      ok, L = bvt.structural.sign_assign(L, S, Rineq, k)
      ok, L = bvt.structural.sign_assign(L, S, Rineq, k, rng)

      L     : n x n candidate impact matrix, np.linalg.cholesky(Sigtilde) @ Q for a rotation
              Q from bvt.structural.qr_sign
      S     : n x m sign restrictions, one column per shock; +1 and -1 restrict the sign of
              that response on impact, NaN leaves it free
      Rineq : m x n x k ranking restrictions, k per shock, each a linear combination of
              impact responses required to be <= 0 for that shock's column; an m x n matrix
              when k = 1. A row of zeros imposes nothing, so np.zeros((m, n)) means sign
              restrictions only
      k     : number of ranking restrictions per shock
      rng   : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      ok    : True when an admissible assignment exists
      L     : on acceptance, column i is the shock-i column, with signs flipped where that is
              what made the restrictions hold and the remaining n-m columns randomly permuted
              and re-signed; unchanged when ok is False

    bvt.structural.sign_restrict requires column i to satisfy shock i, the accept-reject
    scheme of Rubio-Ramirez, Waggoner and Zha (2010). The labeling of the columns of Q is
    arbitrary, so this function accepts whenever every shock has an admissible column, and
    accepts far more often as a result. Proposition 1 of Chan, Matthes and Yu (2026) shows the
    target distribution is unchanged. Both functions stay.

    THE CONDITION. Each shock draws its column with no check that another shock has taken it,
    which is safe only when no column can admit two shocks. That is Assumption 2 of the
    paper, which bvt.structural.check_separable tests; call it once before the rejection
    loop. Restrictions violating it are out of scope: this function raises an error when it
    meets one, and the paper's second algorithm, which enumerates the admissible set, is not
    implemented here.

    THE CALLER must accept or reject the pair (A, Sigma) and Q jointly, since resampling Q
    against a fixed posterior draw targets a different distribution (Arias, Rubio-Ramirez,
    Shin and Waggoner, 2024), and must take one draw per accepted pair, since two assignments
    from the same (Sigma, Q) differ only by a permutation and sign flips.

    On acceptance the rng consumption is one rng.unidrnd per shock, then rng.randperm(n-m)
    and rng.rand(n-m); a rejected candidate consumes nothing.

    See:
    Rubio-Ramirez, J.F., Waggoner, D.F. and Zha, T. (2010). Structural Vector
    Autoregressions: Theory of Identification and Algorithms for Inference, Review of
    Economic Studies, 77(2): 665-696.
    Chan, J.C.C., Matthes, C. and Yu, X. (2026). Large Structural VARs with Multiple Sign and
    Ranking Restrictions, Quantitative Economics, 17(3): 709-740.
    """
    L = np.asarray(L, dtype=float)
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    k = 1 if k is None else int(k)
    R = np.asarray(Rineq, dtype=float).reshape(m, n, k)
        # the m x n table: 1 if column j satisfies every restriction of shock i, -1 if its
        # negative does, 0 otherwise
    pos, neg = (S == 1).astype(float), (S == -1).astype(float)
    nidx = (pos + neg).sum(axis=0)
    up, down = (np.sign(L) == 1).astype(float), (np.sign(L) == -1).astype(float)
    fits = (pos.T @ up + neg.T @ down) == nidx[:, None]
    fits_neg = (pos.T @ down + neg.T @ up) == nidx[:, None]
    rank = np.ones((m, n), dtype=bool)
    rank_neg = np.ones((m, n), dtype=bool)
    for j in range(k):
        rank &= R[:, :, j] @ L <= 0
        rank_neg &= R[:, :, j] @ (-L) <= 0
    satTab = np.where(fits & rank, 1, np.where(fits_neg & rank_neg, -1, 0))

    ok = np.count_nonzero(np.abs(satTab).sum(axis=1)) == m    # admissible set is non-empty
    if not ok:
        return False, L
    r = as_rng(rng)
    L = L.copy()
    reorder = np.zeros(n, dtype=int)
    for i in range(m):
        idx = np.flatnonzero(satTab[i])
        draw = idx[r.unidrnd(idx.size) - 1]
        reorder[i] = draw
        if satTab[i, draw] == -1:
            L[:, draw] = -L[:, draw]
    rest = np.setdiff1d(np.arange(n), reorder[:m])
    if rest.size != n - m:
        raise ValueError('two shocks drew the same column: the restrictions do not separate '
                         'them (see bvt.structural.check_separable)')
    reorder[m:] = rest
    L = L[:, reorder]
        # randomly permute and switch the signs of the last n-m columns
    tmpL2 = L[:, m:][:, r.randperm(n - m)]
    L[:, m:] = tmpL2 * (2 * (r.rand(n - m) > .5) - 1)
    return True, L
