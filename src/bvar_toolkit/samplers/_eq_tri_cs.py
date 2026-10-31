import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def eq_tri_cs(Y, X, XB, B, A, h, tmpdV, beta0, rng=None):
    """bvt.samplers.eq_tri_cs - equation-by-equation Gaussian draw of the VAR
    coefficient matrix B (n x k rows, intercept first) in the CHOLESKY /
    triangularized (CS) SVAR-SV model, conditioning on the unit-lower-triangular
    impact matrix A and the log-volatilities h. Equation ii stacks the rows
    ii:n of the rotated system (Ytilde = Y @ A.T, partialling out the other
    equations' fits through XB @ A.T) and draws B[ii] from its Gaussian
    conditional; XB[:, ii] = X @ betai is refreshed so later equations
    condition on the new draw. Prior variances enter as the stacked k*n vector
    tmpdV, which the caller supplies (the Vbeta output of bvt.priors.vtheta
    produces it); prior means as the stacked k*n vector beta0.

      B, XB = bvt.samplers.eq_tri_cs(Y, X, XB, B, A, h, tmpdV, beta0)
      B, XB = bvt.samplers.eq_tri_cs(Y, X, XB, B, A, h, tmpdV, beta0, rng)

    rng is None, an integer seed, a numpy.random.Generator or a bvt.Stream. The inputs B
    and XB are not modified.

    THIS IS THE CORRECTED TRIANGULAR ALGORITHM of Carriero, Chan, Clark and
    Marcellino (2022), the corrigendum to Carriero, Clark and Marcellino (2019).
    Stacking rows ii:n above is the correction; the original algorithm used equation
    ii alone and so did not draw from the intended conditional.

    See:
    Carriero, A., Chan, J.C.C., Clark, T.E. and Marcellino, M. (2022). Corrigendum
    to: Large Bayesian Vector Autoregressions with Stochastic Volatility and
    Non-Conjugate Priors, Journal of Econometrics, 227(2): 506-512.
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    r = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    XB = np.array(XB, dtype=float)
    B = np.array(B, dtype=float)
    A = np.asarray(A, dtype=float)
    h = np.asarray(h, dtype=float)
    tmpdV = np.asarray(tmpdV, dtype=float).ravel()
    beta0 = np.asarray(beta0, dtype=float).ravel()
    n = Y.shape[1]
    k = X.shape[1]
    Ytilde = Y @ A.T
    for ii in range(n):
        others = np.r_[0:ii, ii + 1:n]
        tmpXBA = XB[:, others] @ A[:, others].T
        Zi = Ytilde[:, ii:] - tmpXBA[:, ii:]
        tmp1 = np.exp(-h[:, ii:]) * A[ii:, ii]
        tmp2 = tmp1 * A[ii:, ii]
        iVbetai = 1 / tmpdV[ii * k:(ii + 1) * k]
        betai0 = beta0[ii * k:(ii + 1) * k]
            # the stacked regressors are X repeated once per equation ii:n
        Kbetai = X.T @ (tmp2.sum(axis=1)[:, None] * X)
        Kbetai[np.diag_indices(k)] += iVbetai
        CKbetai = np.linalg.cholesky(Kbetai)
        rhs = iVbetai * betai0 + X.T @ (tmp1 * Zi).sum(axis=1)
        betai_hat = solve_triangular(CKbetai, solve_triangular(CKbetai, rhs, lower=True),
                                     lower=True, trans='T')
        betai = betai_hat + solve_triangular(CKbetai, r.randn(k), lower=True, trans='T')
        B[ii] = betai
        XB[:, ii] = X @ betai
    return B, XB
