import numpy as np
import scipy.sparse as sp

from .._linalg import BandChol
from .._rng import as_rng


def factor_fsv(Y, X, A, L, h, rng=None):
    """bvt.samplers.factor_fsv - joint Gaussian draw of the whole latent factor path
    F (T x r) in the factor-SV VAR, conditional on the loadings L, the VAR
    coefficients A and the n+r log-volatilities h. Precision sampler on the
    stacked T*r vector; the idiosyncratic variances are exp(h[:, :n]), the factor
    variances exp(h[:, n:]).

      F = bvt.samplers.factor_fsv(Y, X, A, L, h)
      F = bvt.samplers.factor_fsv(Y, X, A, L, h, rng)

      L   : n x r loading matrix (unit-lower-triangular top block)
      h   : T x (n+r), idiosyncratic columns first
      rng : None, an integer seed, a numpy.random.Generator or a bvt.Stream

    See:
    Chan, J.C.C. (2023). Comparing stochastic volatility specifications for large
    Bayesian VARs, Journal of Econometrics, 235(2), 1419-1446.
    """
    rg = as_rng(rng)
    Y = np.asarray(Y, dtype=float)
    L = np.asarray(L, dtype=float).reshape(Y.shape[1], -1)
    h = np.asarray(h, dtype=float)
    T, n = Y.shape
    r = L.shape[1]
    e = (Y - X @ A).ravel()                            # stacked by period
    Xf = sp.kron(sp.eye_array(T), sp.csc_array(L), format='csc')
    XfiSig = Xf.T @ sp.diags_array(np.exp(-h[:, :n]).ravel())
    Kf = sp.diags_array(np.exp(-h[:, n:]).ravel()) + XfiSig @ Xf
    CKf = BandChol(Kf)
    f_hat = CKf.solve_t(CKf.solve(XfiSig @ e))
    f = f_hat + CKf.solve_t(rg.randn(T * r))
    return f.reshape(T, r)
