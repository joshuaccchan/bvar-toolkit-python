import numpy as np
from scipy.linalg import solve_triangular

from .._rng import as_rng


def acp_theta_sig(Y0, Y, p, prior, nsim, rng=None):
    """bvt.samplers.acp_theta_sig - draw the structural VAR parameters under the asymmetric
    conjugate prior, one equation at a time. Each equation's posterior is normal-inverse-gamma
    in closed form.

      Alp, Beta, Sig = bvt.samplers.acp_theta_sig(Y0, Y, p, prior, nsim)
      Alp, Beta, Sig = bvt.samplers.acp_theta_sig(Y0, Y, p, prior, nsim, rng)

      Y0, Y : initial conditions and the estimation sample; the lag matrix is built here from
              them, so no design matrix is passed in
      p     : lag length
      nsim  : number of draws; they are independent, so no burn-in or thinning is needed
      prior : the Struct or dict from bvt.priors.acp_redu or acp_stru (fields beta0, Vbeta,
              alp0, Valp, nu, S)
      rng   : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      Alp   : nsim x n(n-1)/2 free elements of the unit-lower-triangular A
      Beta  : nsim x (n^2 p + n) coefficients, equation by equation
      Sig   : nsim x n structural innovation variances

    Use bvt.structural.reduced_form to map the draws back to the reduced form.

    rng consumption per equation ii, counted from 1, in order: rng.gamrnd(..., nsim) then
    rng.randn(nsim, ki) with ki = n p + ii, filled column by column. One equation's whole block
    is drawn at once, so the stream position after the call depends on nsim as well as on n and
    p.

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169.
    """
    r = as_rng(rng)
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(Y, dtype=float)
    nsim = int(nsim)
    T, n = Y.shape
    tmpY = np.vstack([Y0[-p:], Y])
    Z = np.zeros((T, n * p))
    for ii in range(1, p + 1):
        Z[:, (ii - 1) * n:ii * n] = tmpY[p - ii:tmpY.shape[0] - ii]
    Z = np.hstack([np.ones((T, 1)), Z])
    kb = n * p + 1
    Beta = np.zeros((nsim, n * kb))
    Alp = np.zeros((nsim, n * (n - 1) // 2))
    Sig = np.zeros((nsim, n))
    count_alp = 0
    for ii in range(n):
        yi = Y[:, ii]
        ki = kb + ii
        mi = np.concatenate([prior['beta0'][:, ii], prior['alp0'][count_alp:count_alp + ii]])
        vi = np.concatenate([prior['Vbeta'][:, ii], prior['Valp'][count_alp:count_alp + ii]])
        nui = prior['nu'][ii]
        Si = prior['S'][ii]
        Xi = np.hstack([Z, -Y[:, :ii]])
            # the parameters of the posterior distribution
        ivi = 1.0 / vi
        Ktheta = Xi.T @ Xi
        Ktheta[np.diag_indices(ki)] += ivi
        C = np.linalg.cholesky(Ktheta)
        theta_hat = solve_triangular(C, solve_triangular(C, ivi * mi + Xi.T @ yi, lower=True),
                                     lower=True, trans='T')
        Si_hat = Si + (yi @ yi + (mi * ivi) @ mi - (theta_hat @ Ktheta) @ theta_hat) / 2
            # sample sig and theta
        Sigi = 1.0 / r.gamrnd(nui + T / 2, 1.0 / Si_hat, nsim)
        U = r.randn(nsim, ki) * np.sqrt(Sigi)[:, None]
        Theta = theta_hat[None, :] + solve_triangular(C, U.T, lower=True, trans='T').T   # U / C
        Sig[:, ii] = Sigi
        Beta[:, ii * kb:(ii + 1) * kb] = Theta[:, :kb]
        Alp[:, count_alp:count_alp + ii] = Theta[:, kb:]
        count_alp += ii
    return Alp, Beta, Sig
