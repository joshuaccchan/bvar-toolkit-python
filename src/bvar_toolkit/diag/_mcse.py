import numpy as np

from ._specvar0 import _columns, specvar0


def mcse(draws, L):
    """bvt.diag.mcse - Monte Carlo standard error of the posterior mean estimated from each
    column of a matrix of MCMC draws, the numerical standard error of Geweke (1992): the square
    root of the long-run variance of the sample mean.

      MCSE = bvt.diag.mcse(draws, L)

      draws : R x k, one column per scalar summary (a vector is one chain)
      L     : truncation lag of the Bartlett window, a nonnegative integer below R; take it
              beyond the lag at which the autocorrelations die out
      MCSE  : k-vector

    The long-run variance is estimated by bvt.diag.specvar0. For a textbook discussion, see
    Chan (2027), Section 6.5.2.

    See:
    Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the Calculation
    of Posterior Moments. In: J.M. Bernardo, J.O. Berger, A.P. Dawid and A.F.M. Smith (Eds),
    Bayesian Statistics 4, 169-193, Oxford University Press.
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 6.5.2.
    """
    draws = _columns(draws)
    R, k = draws.shape
    out = np.zeros(k)
    for j in range(k):
        Omega = specvar0(draws[:, j], L)[0] * R    # long-run variance of the draws
        out[j] = np.sqrt(Omega / R)
    return out
