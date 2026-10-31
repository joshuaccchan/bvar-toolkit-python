import numpy as np

from .._linalg import BandChol


def init_approx1N(s2, muh, rhoh, sigh2):
    """bvt.sv.init_approx1N - crude 1-component log-chi2 approximation used to
    initialize SV paths.

      h_hat = bvt.sv.init_approx1N(s2, muh, rhoh, sigh2)

      s2                : T-vector of squared errors
      muh, rhoh, sigh2  : mean, AR(1) coefficient and innovation variance of the
                          log-volatility path
      h_hat             : T-vector, the conditional mean of h under the approximation

    The SV model is approximated by a linear Gaussian state space model in which
    the log chi^2 errors are modeled as N(-1.27,4.94), one normal in place of the
    7-component mixture. Deterministic and deliberately crude: the result is a
    starting value for a sampler.

    See:
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for
    Large Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    ystar = np.log(np.asarray(s2, dtype=float).ravel())
    T = ystar.size
    dv = np.r_[1 - rhoh**2, np.ones(T - 1)]              # HiSH, tridiagonal
    d = dv.copy()
    d[:-1] = dv[:-1] + (-rhoh * dv[1:]) * (-rhoh)
    e = -rhoh * dv[1:]
    HiSH_1 = d.copy()                                    # HiSH @ ones(T)
    HiSH_1[:-1] += e
    HiSH_1[1:] += e
    ab = np.zeros((2, T))
    ab[0] = d / sigh2 + 1 / 4.94
    ab[1, :-1] = e / sigh2
    C = BandChol.from_band(ab if T > 1 else ab[:1])
    return C.solve_t(C.solve(muh / sigh2 * HiSH_1 + (ystar + 1.27) / 4.94))
