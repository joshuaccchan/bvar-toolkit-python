import numpy as np


def irf_redu(A, L, nstep, nshock):
    """bvt.structural.irf_redu - impulse responses of a reduced-form VAR to the structural
    shocks identified by an impact matrix L, computed by powering the companion matrix.

      response = bvt.structural.irf_redu(A, L, nstep, nshock)

      A        : (n p) x n reduced-form coefficients, EXCLUDING the intercept - column ii
                 holds equation ii's coefficients, lag-1 block first
      L        : n x n impact matrix with L @ L.T the reduced-form covariance; the first
                 nshock columns are the identified shocks
      nstep    : number of horizons returned, counting the impact period
      response : n x nshock x nstep; response[:, :, h] holds the responses at horizon h, so
                 response[:, :, 0] = impact

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169.
    """
    A = np.asarray(A, dtype=float)
    L = np.asarray(L, dtype=float)
    nstep, nshock = int(nstep), int(nshock)
    np_, n = A.shape
    p = np_ // n
    response = np.zeros((n, nshock, nstep))
    Acomp = np.zeros((np_, np_))
    Acomp[:n] = A.T
    Acomp[n:, :np_ - n] = np.eye(n * (p - 1))
    response[:, :, 0] = L[:, :nshock]
    Apower = Acomp
    for it in range(1, nstep):
        response[:, :, it] = Apower[:n, :n] @ L[:, :nshock]
        Apower = Apower @ Acomp
    return response
