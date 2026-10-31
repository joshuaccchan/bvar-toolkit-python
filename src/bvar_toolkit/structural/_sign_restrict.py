import numpy as np


def sign_restrict(L, S, Rineq, Ridx):
    """bvt.structural.sign_restrict - test one candidate impact matrix against a set of sign
    restrictions and row inequalities, flipping the sign of any column that satisfies them in
    reverse. This is the accept-reject step of Rubio-Ramirez, Waggoner and Zha (2010).

      ok, L = bvt.structural.sign_restrict(L, S, Rineq, Ridx)

      L     : n x n candidate impact matrix, np.linalg.cholesky(Sigtilde) @ Q for a rotation
              Q from bvt.structural.qr_sign
      S     : n x m sign restrictions, one column per shock. +1 and -1 restrict the sign of
              that response on impact; NaN leaves it free
      Rineq : nR x n rows, each a linear combination required to be NEGATIVE
      Ridx  : nR-vector, the column of L, counted from 0, that each row of Rineq applies to
      ok    : True only if every shock satisfies its sign column AND every row inequality
              holds
      L     : the candidate with columns sign-flipped where that was what made the
              restriction hold. When ok is False, the columns of the shocks tested before the
              first failure keep any flip

    THE ROW INEQUALITIES ARE TESTED STRICTLY: every row must give Rineq[j] @ L[:, Ridx[j]] < 0.
    A row of zeros therefore fails and rejects every candidate, producing an empty identified
    set with no error raised. To impose no ranking restrictions, pass an empty Ridx and a
    0 x n Rineq, NOT a zero row. bvt.structural.sign_assign tests the same quantity as <= 0,
    where a zero row is harmless, so the two differ on exactly this input.

    See:
    Rubio-Ramirez, J.F., Waggoner, D.F. and Zha, T. (2010). Structural Vector
    Autoregressions: Theory of Identification and Algorithms for Inference, Review of
    Economic Studies, 77(2): 665-696.
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169.
    """
    L = np.array(L, dtype=float)
    S = np.asarray(S, dtype=float)
    Ridx = np.asarray(Ridx, dtype=int).ravel()
    Rineq = np.asarray(Rineq, dtype=float).reshape(Ridx.size, L.shape[0])
    m = S.shape[1]
    msat = 0                                        # shocks that satisfy their sign restrictions
    nRsat = 0                                       # row inequalities that hold
    for i in range(m):
        idx = np.flatnonzero((S[:, i] == -1) | (S[:, i] == 1))
        signL = np.sign(L[idx, i])
        if np.all(signL == S[idx, i]):              # column i satisfies the restrictions
            msat += 1
        elif np.all(signL == -S[idx, i]):           # or its negative does
            L[:, i] = -L[:, i]
            msat += 1
        else:
            break
    for j in range(Ridx.size):
        if Rineq[j] @ L[:, Ridx[j]] < 0:
            nRsat += 1
        else:
            nRsat = 0
            break
    return msat == m and nRsat == Ridx.size, L
