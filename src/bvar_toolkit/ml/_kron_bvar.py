import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.special import ndtr

from .._stats import matlab_linspace
from .._struct import Struct
from ._lniwpdf import lniwpdf


def _niw_post(X, Y, w, A0, VA0, S0):
    """The conditional NIW posterior of (A, Sig) under the weights w of the observations: the
    precision KA, the mean Ahat and the scale Shat."""
    XiOm = X.T * w
    KA = XiOm @ X
    KA[np.diag_indices(X.shape[1])] += 1 / VA0
    Ahat = np.linalg.solve(KA, A0 / VA0[:, None] + XiOm @ Y)
    Shat = S0 + A0.T @ (A0 / VA0[:, None]) + (Y.T * w) @ Y - Ahat.T @ KA @ Ahat
    return Ahat, KA, (Shat + Shat.T) / 2            # adjust for rounding errors


def _lmeanexp(x):
    """log(mean(exp(x))) by columns, as max + log(mean(exp(x - max)))."""
    m = np.max(x, axis=0)
    return np.log(np.mean(np.exp(x - m), axis=0)) + m


def _grid(center, lo, hi, n):
    """sort([center; linspace(lo, hi, n)]) and the position of center in it."""
    g = np.sort(np.r_[center, matlab_linspace(lo, hi, n)], kind='stable')
    return g, int(np.flatnonzero(g == center)[0])


def _grid_den(lden, idx, step, center=None):
    """The normalized density at position idx of a grid with log kernel lden; the kernel is
    centered at its maximum, or at position center when given."""
    tmpden = np.exp(lden - (np.max(lden) if center is None else lden[center]))
    tmpden = tmpden / (np.sum(tmpden) * step)
    return tmpden[idx]


def _ltn_const(mu0, V0):
    """log of the normalizing constant 1/(Phi((1-mu0)/s) - Phi((-1-mu0)/s)) of N(mu0, V0)
    truncated to (-1, 1)."""
    return np.log(1 / (ndtr((1 - mu0) / np.sqrt(V0)) - ndtr((-1 - mu0) / np.sqrt(V0))))


def _s2(U, CSig):
    """Row sums of squares of U @ inv(CSig).T."""
    return np.sum(solve_triangular(CSig, U.T, lower=True)**2, axis=0)


def kron_bvar(shortY, X, pri, est):
    """bvt.ml.kron_bvar - log marginal likelihood of the standard (homoskedastic,
    natural-conjugate) BVAR of Chan (2020, JBES), in closed form: the identity
    llike + lpri - lpost, with both NIW ordinates exact, evaluated at A = Ahat,
    the posterior mean, and Sig = Shat/(T+nu0), the inverse of the posterior
    mean of inv(Sig). The identity holds at any point, so the choice of point
    affects only rounding. No rng is consumed.

      ML, out = bvt.ml.kron_bvar(shortY, X, pri, est)

      pri: A0, VA0 (k-vector), nu0, S0
      est: Ahat, Shat, KA, the posterior mean of A, the inverse-Wishart posterior scale
           and the posterior precision of each column of A
      out: llike, and the evaluation point A, Sig

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    shortY = np.asarray(shortY, dtype=float)
    X = np.asarray(X, dtype=float)
    T, n = shortY.shape
    A0, VA0 = np.asarray(pri['A0'], dtype=float), np.asarray(pri['VA0'], dtype=float).ravel()
    nu0, S0 = pri['nu0'], np.asarray(pri['S0'], dtype=float)
    Ahat, Shat, KA = (np.asarray(est['Ahat'], dtype=float), np.asarray(est['Shat'], dtype=float),
                      est['KA'])
    A = Ahat
    Sig = Shat / (T + nu0)
    CSig = np.linalg.cholesky(Sig)
    s2 = _s2(shortY - X @ A, CSig)
    llike = -T * n / 2 * np.log(2 * np.pi) - T * np.sum(np.log(np.diag(CSig))) - .5 * np.sum(s2)
    ML = llike + lniwpdf(A, Sig, A0, sp.diags_array(1 / VA0), nu0, S0) \
        - lniwpdf(A, Sig, Ahat, KA, nu0 + T, Shat)
    return ML, Struct(llike=llike, A=A, Sig=Sig)
