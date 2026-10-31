import numpy as np

from .._rng import as_rng


def svo_outlier(Y, X, A, B0, h, o_grid, po, p0a, p0b, rng=None):
    """bvt.sv.svo_outlier - conditional draw of the outlier scales o_t (from a
    discrete grid, one independent draw per period) and of the outlier
    probability po (beta conjugate) in the VAR-SVO model. Period t is an outlier
    when o_t > 1; the grid's first point is 1, the remaining ngrid points share
    the prior mass po.

      o, po = bvt.sv.svo_outlier(Y, X, A, B0, h, o_grid, po, p0a, p0b)
      o, po = bvt.sv.svo_outlier(Y, X, A, B0, h, o_grid, po, p0a, p0b, rng)

      Y, X   : T x n data and T x k regressors
      A      : k x n VAR coefficient matrix
      B0     : n x n impact matrix; (Y - X @ A) @ B0.T are the orthogonalized errors
      h      : T x n log-volatility paths
      o_grid : (ngrid+1)-vector, [1, grid of outlier sizes]
      po     : current outlier probability; the new draw is returned
      p0a,p0b: beta prior parameters
      rng    : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      o      : T-vector draw, each element one of the o_grid points

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    o_grid = np.asarray(o_grid, dtype=float).ravel()
    T, n = Y.shape
    ngrid = o_grid.size - 1
    o_lpri = np.log(np.r_[1 - po, np.repeat(po / ngrid, ngrid)])
    U = ((Y - X @ A) @ np.asarray(B0, dtype=float).T) / np.exp(np.asarray(h, dtype=float) / 2)
    uu = np.sum(U * U, axis=1)
    lliket = -n * np.log(o_grid) - .5 * uu[:, None] / o_grid**2
    o_post = np.exp(lliket + o_lpri - lliket.max(axis=1, keepdims=True))
    o_post = o_post / o_post.sum(axis=1, keepdims=True)
    u = r.rand(T)                                        # one uniform per period, in order
    o = o_grid[np.argmax(u[:, None] < np.cumsum(o_post, axis=1), axis=1)]

        # sample po
    tmp = np.sum(o > 1)
    po = r.betarnd(p0a + tmp, p0b + T - tmp)
    return o, float(po)
