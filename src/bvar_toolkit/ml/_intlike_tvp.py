import numpy as np
import scipy.sparse as sp

from .._linalg import BandChol


def _rw_prec(inv_var, T, k):
    """H' diag(repeated inv_var) H for the random walk of k states over T periods, H the
    first difference matrix of the stacked states."""
    H = (sp.eye_array(T * k, format='csc') - sp.eye_array(T * k, k=-k, format='csc')).tocsc()
    return (H.T @ sp.diags_array(np.tile(inv_var, T)) @ H).tocsc()


def intlike_tvp(Y, Sig, Sigtheta, bigX, theta0):
    """bvt.ml.intlike_tvp - log integrated likelihood of a TVP-VAR with constant error
    variances, the states theta_t = theta_{t-1} + u_t integrated out analytically.

      llike = bvt.ml.intlike_tvp(Y, Sig, Sigtheta, bigX, theta0)

      Y        : T*n data, stacked by period
      Sig      : n error variances
      Sigtheta : k variances of the state innovations
      bigX     : T*n x T*k regressor matrix of the stacked states
      theta0   : k initial state

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Sig = np.asarray(Sig, dtype=float).ravel()
    Sigtheta = np.asarray(Sigtheta, dtype=float).ravel()
    theta0 = np.asarray(theta0, dtype=float).ravel()
    bigX = sp.csc_array(bigX)
    n = Sig.size
    T = Y.size // n
    k = Sigtheta.size
    iSig = np.tile(1 / Sig, T)
    XinvSig = bigX.T @ sp.diags_array(iSig)
    HinvSH = _rw_prec(1 / Sigtheta, T, k)
    alptheta = np.tile(theta0, T)
    Ktheta = (HinvSH + XinvSig @ bigX).tocsc()
    dtheta = XinvSig @ Y + HinvSH @ alptheta
    CKtheta = BandChol(Ktheta)
    return -T * n / 2 * np.log(2 * np.pi) - T / 2 * np.sum(np.log(Sigtheta)) \
        - T / 2 * np.sum(np.log(Sig)) - CKtheta.logdet() / 2 \
        - .5 * (Y @ (iSig * Y) + alptheta @ (HinvSH @ alptheta)
                - dtheta @ CKtheta.solve_t(CKtheta.solve(dtheta)))
