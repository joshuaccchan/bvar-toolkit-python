import numpy as np
from scipy.special import digamma, polygamma


def dirifit(X):
    """bvt.ml.dirifit - maximum likelihood fit of a Dirichlet distribution to the rows
    of X, by Newton's method started at the column means.

      alphat, flag = bvt.ml.dirifit(X)

    X is N x k with rows in the simplex; alphat is a k-vector. flag is 0 when an iterate
    turns negative, where the fit stops; otherwise 1. The iteration stops once any
    element of the Newton step is at most 1e-4 in absolute value. This is the
    published stopping rule, kept deliberately: the importance densities of the RS
    marginal likelihoods depend on it.

    See:
    Chan, J.C.C. and Eisenstat, E. (2018). Bayesian model comparison for
    time-varying parameter VARs with stochastic volatility, Journal of Applied
    Econometrics, 33(4), 509-532.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    alphat = X.mean(axis=0)
    logX = np.log(X)
    err = np.ones(1)
    flag = 1
    while np.all(np.abs(err) > 1e-4):
        if np.min(alphat) < 0:
            flag = 0
            break
        S = np.sum(digamma(np.sum(alphat)) - digamma(alphat) + logX, axis=0)
        H = N * polygamma(1, np.sum(alphat)) - N * np.diag(polygamma(1, alphat))
        err = -np.linalg.solve(H, S)
        alphat = alphat + err
    return alphat, flag
