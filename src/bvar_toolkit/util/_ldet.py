import numpy as np
import scipy.sparse as sp


def ldet(Omega):
    """bvt.util.ldet - log determinant of a symmetric positive definite matrix, taken from its
    Cholesky factor.

      k = bvt.util.ldet(Omega)

      Omega : symmetric positive definite matrix; numpy.linalg.cholesky raises LinAlgError if
              it is not. A sparse matrix is factored as a dense one
      k     : log(det(Omega))

    See:
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for Large Bayesian
    VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    if sp.issparse(Omega):
        Omega = Omega.toarray()
    C = np.linalg.cholesky(np.asarray(Omega, dtype=float))
    return 2 * np.sum(np.log(np.diag(C)))
