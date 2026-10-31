import numpy as np
import scipy.sparse as sp


def _mvnpdf_diag(X, mu, sig2):
    """Normal densities of the rows of X with means mu and diagonal variances sig2, as MATLAB's
    mvnpdf computes them for a diagonal covariance given as a vector."""
    R = np.sqrt(sig2)
    quadform = np.sum(((X - mu) / R)**2, axis=1)
    return np.exp(-0.5 * quadform - np.sum(np.log(R)) - X.shape[1] * np.log(2 * np.pi) / 2)


def intlike_rsvar(shortY, bigX, theta, Sig, P, p1=None):
    """bvt.ml.intlike_rsvar - log likelihood of a regime-switching VAR with Markov
    regimes, the regimes integrated out by the Hamilton filter.

      llike = bvt.ml.intlike_rsvar(shortY, bigX, theta, Sig, P)
      llike = bvt.ml.intlike_rsvar(shortY, bigX, theta, Sig, P, p1)

      shortY : T x n data
      bigX   : T*n x k regressor matrix, bigX @ theta[:, j] the stacked conditional
               mean in regime j
      theta  : k x r coefficients, one column per regime
      Sig    : n x r error variances, one column per regime
      P      : r x r transition matrix, P[i, j] = p(s_t = j | s_{t-1} = i)
      p1     : p(s_1 = j), the same for every j; default 1/r. The published routine
               uses 1/3 whatever r is, which shifts the log likelihood by log(r/3);
               p1 = 1/3 reproduces it.

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    shortY = np.asarray(shortY, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(np.shape(theta)[0], -1)
    Sig = np.asarray(Sig, dtype=float).reshape(np.shape(Sig)[0], -1)
    P = np.atleast_2d(np.asarray(P, dtype=float))
    bigX = bigX if sp.issparse(bigX) else np.asarray(bigX, dtype=float)
    r = theta.shape[1]
    T, n = shortY.shape
    if p1 is None:
        p1 = 1 / r
    llike = 0.0
    like = np.zeros((T, r))
    tmpP1 = np.zeros((T, r))                     # p(s_t|Y_t,theta,P)
    tmpP2 = np.zeros((T, r))                     # p(s_t|Y_{t-1},theta,P)
    tmpP2[0] = p1

        # compute filtering probabilities
    for i in range(r):
        mu_i = bigX @ theta[:, i]
        like[:, i] = _mvnpdf_diag(shortY, mu_i.reshape(T, n), Sig[:, i])
    for t in range(T):
        if t > 0:
            tmpP2[t] = tmpP1[t - 1] @ P
        tmp = tmpP2[t] * like[t]
        tmpP1[t] = tmp / np.sum(tmp)
        llike = llike + np.log(np.sum(tmp))
    return llike
