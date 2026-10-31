import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.special import gammaln


def lniwpdf(A, Sig, A0, iVA0, nu0, S0):
    """bvt.ml.lniwpdf - log density of the normal-inverse-Wishart distribution at
    (A,Sig): Sig ~ IW(nu0,S0), vec(A)|Sig ~ N(vec(A0), Sig kron inv(iVA0)) - the
    Kronecker-structured conjugate form, with iVA0 the PRECISION (inverse prior
    covariance) of each column's coefficients.

    Inputs:  A    - k x n evaluation point (VAR coefficients)
             Sig  - n x n evaluation point (error covariance)
             A0   - k x n mean matrix
             iVA0 - k x k precision matrix (sparse or dense): pass
                    sp.diags_array(1/VA0) for a prior ordinate, the posterior
                    precision KA for a posterior ordinate
             nu0  - IW degrees of freedom
             S0   - n x n IW scale matrix
    Output:  lden - log density

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    A = np.asarray(A, dtype=float)
    Sig = np.asarray(Sig, dtype=float)
    S0 = np.asarray(S0, dtype=float)
    iVA0 = iVA0.toarray() if sp.issparse(iVA0) else np.asarray(iVA0, dtype=float)
    k, n = A.shape
    CSig = np.linalg.cholesky(Sig)
    cA = -n * k / 2 * np.log(2 * np.pi) + n * np.sum(np.log(np.diag(np.linalg.cholesky(iVA0))))
    cSig = -nu0 * n / 2 * np.log(2) - n * (n - 1) / 4 * np.log(np.pi) \
        - np.sum(gammaln((nu0 + 1 - np.arange(1, n + 1)) / 2)) \
        + nu0 * np.sum(np.log(np.diag(np.linalg.cholesky(S0))))
    tmp = A - np.asarray(A0, dtype=float)
    M = S0 + tmp.T @ iVA0 @ tmp
    W = solve_triangular(CSig, solve_triangular(CSig, M, lower=True), lower=True, trans='T')
    return cA + cSig - (n + nu0 + k + 1) * np.sum(np.log(np.diag(CSig))) - .5 * np.trace(W)
