import numpy as np


def logsumexp(x, axis=0):
    """bvt.util.logsumexp - numerically stable log(sum(exp(x))) along an axis.

      y = bvt.util.logsumexp(x)        operates along axis 0
      y = bvt.util.logsumexp(x, axis)

      x    : array of log-scale values; a slice that is entirely -inf returns -inf, with no NaN.
             A 1-D array is one column
      axis : axis to reduce, counted from 0 (default 0)

    For averaging M log predictive-likelihood draws, use
      logsumexp(logdraws) - log(M)
    """
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    ms = np.where(np.isfinite(m), m, 0.0)            # avoid inf - inf = NaN below
    with np.errstate(divide='ignore'):
        y = ms + np.log(np.sum(np.exp(x - ms), axis=axis, keepdims=True))
    y = np.where(m == -np.inf, -np.inf, y)           # all -inf slices
    y = np.squeeze(y, axis=axis)
    return y[()] if y.ndim == 0 else y
