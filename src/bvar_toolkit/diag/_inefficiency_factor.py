import numpy as np

from ._specvar0 import _columns, specvar0


def inefficiency_factor(draws, L):
    """bvt.diag.inefficiency_factor - inefficiency factor of each column of a matrix of MCMC
    draws: the long-run variance of the draws over their variance, which is the reciprocal of
    the relative numerical efficiency of Geweke (1992). It is the number of draws that carry the
    information of one independent draw.

      IF = bvt.diag.inefficiency_factor(draws, L)

      draws : R x k, one column per scalar summary (a vector is one chain)
      L     : truncation lag of the Bartlett window, a nonnegative integer below R; take it
              beyond the lag at which the autocorrelations die out
      IF    : k-vector; the effective sample sizes are R/IF

    The long-run variance is estimated by bvt.diag.specvar0. For a textbook discussion, see
    Chan (forthcoming), Section 6.5.2.

    See:
    Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the Calculation
    of Posterior Moments. In: J.M. Bernardo, J.O. Berger, A.P. Dawid and A.F.M. Smith (Eds),
    Bayesian Statistics 4, 169-193, Oxford University Press.
    Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 6.5.2.
    """
    draws = _columns(draws)
    R, k = draws.shape
    IF = np.zeros(k)
    for j in range(k):
        x = draws[:, j]
        sigma2 = np.var(x)                          # variance, divisor R
        Omega = specvar0(x, L)[0] * R               # long-run variance of the draws
        IF[j] = Omega / sigma2
    return IF
