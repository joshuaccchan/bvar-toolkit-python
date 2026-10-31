import numpy as np

from .._linalg import ma_solve
from .._rng import as_rng
from ._intlike_csv import _em_t, _hish, _s2, _weights
from ._intlike_t_csv import _deny_t


def intlike_csv_t_ma(shortY, X, A, Sig, psi, rho, sigh2, nu, R, rng=None):
    """bvt.ml.intlike_csv_t_ma - integrated likelihood of the BVAR-CSV-t-MA model
    at (A,Sig,psi,rho,sigh2,nu): the Student-t scales lam are integrated
    analytically and the common log-volatility path h by importance sampling
    (R draws), with the EM-within-Newton mode search of the t-CSV case run on
    the MA(1)-transformed residuals Utld = Hpsi^{-1} U.

      intlike, store_llike = bvt.ml.intlike_csv_t_ma(shortY, X, A, Sig, psi, rho,
                                                    sigh2, nu, R)
      intlike, store_llike = bvt.ml.intlike_csv_t_ma(..., rng)

    rng is None, an integer seed, a numpy.random.Generator or a bvt.Stream.

    Known quirk, kept in both modes of bvt.ml.kron_bvar_csv_t_ma: unlike
    bvt.ml.intlike_csv_ma, this function treats the first transformed observation
    as t with scale exp(h_1)*Sig, with no (1+psi^2) factor and no
    -n/2*log(1+psi^2) term, whereas the Gibbs sampler it pairs with gives it
    variance (1+psi^2)*exp(h_1)*lam_1*Sig. The published BVAR-CSV-t-MA marginal
    likelihood includes this one-observation mismatch.

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    r = as_rng(rng)
    shortY = np.asarray(shortY, dtype=float)
    T, n = shortY.shape
    HiSH = _hish(rho, sigh2, T)

        # obtain the mode and negative Hessian of the conditional density of h
    CSig = np.linalg.cholesky(np.asarray(Sig, dtype=float))
    U = shortY - np.asarray(X, dtype=float) @ np.asarray(A, dtype=float)
    s2 = _s2(ma_solve(psi, U), CSig)
    ht, Kh, CKh = _em_t(s2, n, nu, HiSH, np.log(np.mean(s2)) * np.ones(T))

        # evaluate the importance weights
    return _weights(ht, Kh, CKh, HiSH, rho, sigh2, int(R), r, _deny_t(s2, CSig, n, nu))
