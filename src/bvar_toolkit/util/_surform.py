import numpy as np
import scipy.sparse as sp


def surform(X):
    """bvt.util.surform - T x Tk BLOCK-DIAGONAL sparse SUR expansion (TVP state stacking).

      Xout = bvt.util.surform(X)

      X    : T x k matrix of regressors
      Xout : T x T*k sparse matrix (csc_array) whose row t holds X[t] in columns t*k to
             t*k + k - 1, so that Xout @ [b_1; b_2; ...; b_T] returns the fitted series of the
             TVP regression y_t = X[t] @ b_t

    NOT the same operator as bvt.util.surform2, which is the n-row Kronecker expansion
    kron(I_n, X[t]) for stacked-vector VARs.

    See:
    Chan, J.C.C. (2023). Large Hybrid Time-Varying Parameter VARs, Journal of Business and
    Economic Statistics, 41(3): 890-905
    """
    X = np.asarray(X, dtype=float)
    r, c = X.shape
    Xout = sp.csc_array((X.reshape(-1), (np.repeat(np.arange(r), c), np.arange(r * c))),
                        shape=(r, r * c))
    Xout.eliminate_zeros()
    return Xout
