import numpy as np


def kron_gram(X, B0, W, U):
    """bvt.util.kron_gram - weighted Gram matrix of kron(B0, X), formed from its k x k blocks.
    With Xtilde = kron(B0, X) * sqrt(vec(W))[:, None] and ytilde = vec(U) * sqrt(vec(W)), it
    returns Xtilde.T @ Xtilde, Xtilde.T @ ytilde and ytilde @ ytilde without forming Xtilde, in
    O(nTk^2 + n^3k^2) operations against O(Tn^3k^2).

      XWX, XWy, yWy = bvt.util.kron_gram(X, B0, W, U)

      X   : T x k regressors
      B0  : n x n matrix
      W   : T x n weights, W[t, i] the inverse variance of equation i at time t
      U   : T x n, typically Y @ B0.T
      XWX : nk x nk; block (j, l) is the sum over i of B0[i, j]*B0[i, l]*X.T @ diag(W[:, i]) @ X
      XWy : nk-vector, vec(X.T @ (U*W) @ B0)
      yWy : np.sum(U**2 * W)
    """
    X = np.asarray(X, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)
    k = X.shape[1]
    n = B0.shape[0]
    G = np.zeros((k * k, n))
    for i in range(n):
        G[:, i] = (X.T @ (X * W[:, [i]])).reshape(k * k, order='F')     # X' diag(W[:, i]) X
    Cf = np.zeros((n, n * n))                       # Cf[i, j*n + l] = B0[i, j]*B0[i, l]
    for j in range(n):
        Cf[:, j * n:(j + 1) * n] = B0[:, [j]] * B0
    XWX = np.transpose((G @ Cf).reshape((k, k, n, n), order='F'),
                       (0, 3, 1, 2)).reshape((n * k, n * k), order='F')
    XWy = (X.T @ (U * W) @ B0).reshape(n * k, order='F')
    yWy = np.sum(U ** 2 * W)
    return XWX, XWy, yWy
