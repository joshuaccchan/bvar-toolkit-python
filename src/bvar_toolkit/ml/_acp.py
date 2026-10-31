import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln


def acp(p, Y, Z, prior, ridge=0.0):
    """bvt.ml.acp - log marginal likelihood of a VAR under the asymmetric conjugate prior, in
    CLOSED FORM.

      lml = bvt.ml.acp(p, Y, Z, prior)
      lml = bvt.ml.acp(p, Y, Z, prior, ridge=1e-6)

      Z     : T x (n p + 1) lag matrix, intercept first (bvt.util.build_lags)
      prior : the Struct or dict from bvt.priors.acp_redu or acp_stru
      ridge : ridge*I added to the posterior precision inv(Vi) + Xi.T @ Xi of every equation;
              default 0. A positive value changes the log marginal likelihood by enough to
              affect a model comparison, so hold it fixed across the models being compared

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169.
    """
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    T, n = Y.shape
    lml = -n * T / 2 * np.log(2 * np.pi)
    count_alp = 0
    for ii in range(n):
        yi = Y[:, ii]
        ki = n * p + ii + 1
        mi = np.concatenate([prior['beta0'][:, ii], prior['alp0'][count_alp:count_alp + ii]])
        vi = np.concatenate([prior['Vbeta'][:, ii], prior['Valp'][count_alp:count_alp + ii]])
        nui = prior['nu'][ii]
        Si = prior['S'][ii]
        Xi = np.hstack([Z, -Y[:, :ii]])
        ivi = 1.0 / vi                              # the diagonal of inv(Vi)
        Ktheta = Xi.T @ Xi
        Ktheta[np.diag_indices(ki)] += ivi
        if ridge != 0:
            Ktheta[np.diag_indices(ki)] += ridge
        C = np.linalg.cholesky(Ktheta)
        theta_hat = solve_triangular(C, solve_triangular(C, ivi * mi + Xi.T @ yi, lower=True),
                                     lower=True, trans='T')
        Si_hat = Si + (yi @ yi + (mi * ivi) @ mi - (theta_hat @ Ktheta) @ theta_hat) / 2
        lml = (lml - 1 / 2 * (np.sum(np.log(vi)) + 2 * np.sum(np.log(np.diag(C))))
               + nui * np.log(Si) - (nui + T / 2) * np.log(Si_hat) + gammaln(nui + T / 2) - gammaln(nui))
        count_alp += ii
    return lml
