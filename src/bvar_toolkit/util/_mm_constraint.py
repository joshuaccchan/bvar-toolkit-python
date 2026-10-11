import numpy as np
import scipy.sparse as sp


def mm_constraint(X, isq, weights=None):
    """bvt.util.mm_constraint - the linear restrictions M @ y = z that tie quarterly growth
    rates to the monthly growth rates of the same series, by the log-linear aggregation of
    Mariano and Murasawa (2003).

      M, z, Y, dropped = bvt.util.mm_constraint(X, isq)
      M, z, Y, dropped = bvt.util.mm_constraint(X, isq, weights=w)

      X       : T x n monthly data. A quarterly column holds the quarterly growth rate, 100
                times the log change, in the last month of each quarter and NaN elsewhere
      isq     : boolean n-vector, True for the quarterly columns
      weights : w[j] multiplies the monthly value j months before the quarterly one; default
                [1, 2, 3, 2, 1]/3, so z_t = (y_t + 2y_{t-1} + 3y_{t-2} + 2y_{t-3} + y_{t-4})/3
      M       : q x Tn sparse csc_array, one row per quarterly value, acting on
                y = vec(Y') = Y.ravel(), the stacking of bvt.util.select_obs
      z       : q-vector of quarterly values
      Y       : X with every entry of the quarterly columns set to NaN, the monthly values to be
                drawn
      dropped : T x n booleans, the quarterly values left out because their window starts
                before the first month

    The monthly unknowns are monthly growth rates in percent per month. The monthly columns pass
    through unchanged, NaN included. Written for this toolkit.

    See:
    Mariano, R.S. and Murasawa, Y. (2003). A New Coincident Index of Business Cycles Based on
    Monthly and Quarterly Series, Journal of Applied Econometrics, 18(4): 427-443.
    Chan, J.C.C., Poon, A. and Zhu, D. (2023). High-Dimensional Conditionally Gaussian State
    Space Models with Missing Data, Journal of Econometrics, 236(1): 105468, Section 2.2.
    """
    w = np.array([1, 2, 3, 2, 1]) / 3 if weights is None else np.asarray(weights, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.size == 0:
        raise ValueError('X must be a nonempty T x n numeric matrix')
    T, n = X.shape
    isq = np.asarray(isq, dtype=bool).ravel()
    if isq.size != n or not isq.any():
        raise ValueError(f'isq must hold {n} booleans with at least one True')
    if w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError('the weights must be a nonempty finite vector')
    L = w.size

    rows, cols, vals, z = [], [], [], []
    dropped = np.zeros((T, n), dtype=bool)
    q = 0
    for i in np.flatnonzero(isq):
        for t in np.flatnonzero(~np.isnan(X[:, i])):
            if t < L - 1:
                dropped[t, i] = True
                continue
            s = t - np.arange(L)                    # months t, t-1, ..., t-L+1
            rows += [q] * L
            cols += list(s * n + i)
            vals += list(w)
            z.append(X[t, i])
            q += 1
    M = sp.csc_array((np.array(vals, dtype=float),
                      (np.array(rows, dtype=int), np.array(cols, dtype=int))), shape=(q, T * n))
    Y = X.copy()
    Y[:, isq] = np.nan
    return M, np.array(z, dtype=float), Y, dropped
