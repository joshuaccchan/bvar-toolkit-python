import numpy as np
from scipy.special import ndtr

from .._struct import Struct
from ._specvar0 import _columns, specvar0


def geweke(draws, a=0.10, b=0.50, L='auto'):
    """bvt.diag.geweke - the convergence diagnostic of Geweke (1992) for each column of a matrix
    of MCMC draws: the difference between the means of an early and a late segment of the chain,
    over the standard error implied by the long-run variances of the two segment means.

      Z, pval, info = bvt.diag.geweke(draws)
      Z, pval, info = bvt.diag.geweke(draws, a, b, L)

      draws : R x k, one column per scalar summary (a vector is one chain)
      a, b  : the early segment is draws 1 to floor(a*R) and the late segment draws floor(b*R)
              to R, with 0 < a < b < 1 (defaults .1 and .5)
      L     : truncation lag of the Bartlett window in both segments; the default, 'auto', takes
              floor(4*(n/100)^(2/9)) for a segment of n draws, the rule of thumb of Newey and
              West (1994)
      Z     : k-vector, approximately standard normal if the chain is stationary
      pval  : k-vector of two-sided p-values
      info  : Struct with the segments A and B (row indices of draws, counted from 0), their
              lags LA and LB, and a and b

    The long-run variances are estimated by bvt.diag.specvar0. For a textbook discussion, see
    Chan (forthcoming), Section 6.5.1.

    See:
    Geweke, J. (1992). Evaluating the Accuracy of Sampling-Based Approaches to the Calculation
    of Posterior Moments. In: J.M. Bernardo, J.O. Berger, A.P. Dawid and A.F.M. Smith (Eds),
    Bayesian Statistics 4, 169-193, Oxford University Press.
    Newey, W.K. and West, K.D. (1994). Automatic Lag Selection in Covariance Matrix Estimation,
    Review of Economic Studies, 61(4): 631-653.
    Chan, J.C.C. (forthcoming). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC, Section 6.5.1.
    """
    draws = _columns(draws)
    R, k = draws.shape
    if not (np.isscalar(a) and np.isscalar(b) and 0 < a < b < 1):
        raise ValueError('a and b must satisfy 0 < a < b < 1')
    A = np.arange(int(np.floor(a * R)))
    B = np.arange(int(np.floor(b * R)) - 1, R)
    nA, nB = A.size, B.size
    if isinstance(L, str):
        if L.lower() != 'auto':
            raise ValueError("L must be 'auto' or a nonnegative integer")
        LA = max(0, int(np.floor(4 * (nA / 100) ** (2 / 9))))
        LB = max(0, int(np.floor(4 * (nB / 100) ** (2 / 9))))
    else:
        LA = max(0, int(np.floor(L)))
        LB = LA
    if nA < 2 or LA >= nA or LB >= nB:
        raise ValueError(f'the segments have {nA} and {nB} draws, too few for lags {LA} and {LB}')
    Z = np.full(k, np.nan)
    pval = np.full(k, np.nan)
    for j in range(k):
        xA = draws[A, j]
        xB = draws[B, j]
        SA = specvar0(xA, LA)[0]                    # long-run variance of the mean of A
        SB = specvar0(xB, LB)[0]                    # and of B
        Z[j] = (np.mean(xA) - np.mean(xB)) / np.sqrt(SA + SB)
        pval[j] = 2 * (1 - ndtr(abs(Z[j])))
    return Z, pval, Struct(A=A, B=B, LA=LA, LB=LB, a=a, b=b)
