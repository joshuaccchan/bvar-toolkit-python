import numpy as np
from scipy.linalg import solve_triangular, svd

from .._rng import as_rng
from ..util import anormrnd


def b0_row_sampler(U, h, B0, B00, VB0, rng=None):
    """bvt.structural.b0_row_sampler - one full row-wise pass of the posterior draw
    of the impact matrix B0 in the order-invariant SVAR-SV model, rows ii = 1:n in
    order, each row returned with a positive ii-th element. The row-by-row
    construction is that of Waggoner and Zha (2003), with the extension to a
    nonzero prior mean and the two-component approximation to the absolute-normal
    density of Villani (2009), drawn in bvt.util.anormrnd.

      B0 = bvt.structural.b0_row_sampler(U, h, B0, B00, VB0)
      B0 = bvt.structural.b0_row_sampler(U, h, B0, B00, VB0, rng)

      U   : T x n residual matrix, formed by the CALLER - Y - X @ A when
            estimating, Yt - Xt @ A when forecasting
      h   : T x n log-volatilities
      B0  : current n x n impact matrix (rows updated in order; the input is not
            modified)
      B00 : n x n prior mean of B0   (eye(n) in the paper)
      VB0 : n x n prior variances    (ones((n, n)) in the paper)
      rng : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    rng consumption, per row: one rng.rand then one rng.randn (inside anormrnd), then
    n-1 further rng.randn - rows in order ii = 1:n.

    See:
    Waggoner, D.F. and Zha, T. (2003). A Gibbs Sampler for Structural Vector
    Autoregressions, Journal of Economic Dynamics and Control, 28(2): 349-366.
    Villani, M. (2009). Steady-State Priors for Vector Autoregressions, Journal
    of Applied Econometrics, 24(4): 630-650.
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    r = as_rng(rng)
    U = np.asarray(U, dtype=float)
    h = np.asarray(h, dtype=float)
    B0 = np.array(B0, dtype=float)
    B00 = np.asarray(B00, dtype=float)
    VB0 = np.asarray(VB0, dtype=float)
    T, n = U.shape
    for ii in range(n):
        EiOhi = U.T * np.exp(-h[:, ii])
        Kbi = EiOhi @ U
        Kbi[np.diag_indices(n)] += 1 / VB0[ii]
        CKbi = np.linalg.cholesky(Kbi)
        mui = solve_triangular(CKbi, solve_triangular(CKbi, B00[ii] / VB0[ii], lower=True),
                               lower=True, trans='T')
        Ci = CKbi / np.sqrt(T)
        Gam_mi = np.delete(B0, ii, axis=0).T
        Gam_miperp = _null(Gam_mi.T)

        V = np.zeros((n, n))
        zeta = np.zeros(n)
        for jj in range(n):
            if jj == 0:
                v1 = solve_triangular(Ci, Gam_miperp, lower=True)
                v1 = v1 / np.linalg.norm(v1)
                V = np.hstack([v1, _null(v1.T)])
                zetaj_hat = mui @ (Ci @ v1[:, 0])
                zeta[0] = anormrnd(zetaj_hat, 1 / T, rng=r)
            else:
                zetaj_hat = mui @ (Ci @ V[:, jj])
                zeta[jj] = zetaj_hat + 1 / np.sqrt(T) * r.randn()
        phii = solve_triangular(Ci, V @ zeta, lower=True, trans='T')
        B0[ii] = phii * np.sign(phii[ii])  # fix the sign of the ii-th element to be positive
    return B0


def _null(A):
    """An orthonormal basis of the null space of A, as MATLAB's null(A) computes it: the left
    singular vectors of A.T beyond the numerical rank, from LAPACK's gesvd, whose signs agree
    with MATLAB's. The signs enter the draw."""
    m, n = A.shape
    U, s, _ = svd(A.T, lapack_driver='gesvd')
    tol = max(m, n) * np.spacing(s.max()) if s.size else 0.0
    return U[:, int(np.sum(s > tol)):]
