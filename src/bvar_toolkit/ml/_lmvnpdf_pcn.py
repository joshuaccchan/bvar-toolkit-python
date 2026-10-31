import numpy as np
import scipy.sparse as sp

from .._linalg import BandChol


def lmvnpdf_pcn(x, mu, K):
    """bvt.ml.lmvnpdf_pcn - log density of N(mu, inv(K)) at x, parameterized by the
    PRECISION K (not the covariance).

      lden = bvt.ml.lmvnpdf_pcn(x, mu, K)

    K may be dense or a sparse banded matrix.

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    x = np.asarray(x, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    n = mu.size
    if sp.issparse(K):
        C = BandChol(K)
        ldiag = np.log(C.ab[0])
        v = x - mu                                  # e = C' (x - mu), as the band product
        e = C.ab[0] * v
        for d in range(1, C.ab.shape[0]):
            e[:-d] += C.ab[d, :-d] * v[d:]
    else:
        CK = np.linalg.cholesky(np.asarray(K, dtype=float))
        ldiag = np.log(np.diag(CK))
        e = CK.T @ (x - mu)
    return -n / 2 * np.log(2 * np.pi) + np.sum(ldiag) - .5 * (e @ e)
