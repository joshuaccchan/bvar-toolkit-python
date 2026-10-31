import numpy as np
from scipy.special import digamma, gammaln, polygamma

from .._rng import as_rng


def nu_studentt(lam, nu, nu_ub, rng=None):
    """bvt.sv.nu_studentt - independence-MH draw of the Student-t degree-of-freedom
    parameter nu given the latent mixture scales lam (lam_t ~ IG(nu/2, nu/2)),
    under a uniform prior on (2, nu_ub): Newton-Raphson mode nut, Gaussian
    N(nut, -1/H) proposal, MH accept-reject.

      nu, flag, f_nu = bvt.sv.nu_studentt(lam, nu, nu_ub)
      nu, flag, f_nu = bvt.sv.nu_studentt(lam, nu, nu_ub, rng)

    Inputs:  lam   - T-vector of latent scale draws
             nu    - current df draw
             nu_ub - upper bound of the uniform prior support
             rng   - None, an integer seed, a numpy.random.Generator or a bvt.Stream
    Outputs: nu, flag (1 if accepted), f_nu (the log target kernel of nu, a function)

    nu comes back unchanged with flag = 0 both when the candidate is rejected and
    when it falls outside (2, nu_ub); flag marks an accepted move.

    See:
    Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds),
    Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham.
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    r = as_rng(rng)
    lam = np.asarray(lam, dtype=float).ravel()
    flag = 0
    T = lam.size
    sum1 = np.sum(np.log(lam))
    sum2 = np.sum(1 / lam)

    def f_nu(x):
        return T * (x / 2 * np.log(x / 2) - gammaln(x / 2)) - (x / 2 + 1) * sum1 - x / 2 * sum2

    def df_nu(x):
        return T / 2 * (np.log(x / 2) + 1 - digamma(x / 2)) - .5 * (sum1 + sum2)

    def d2f_nu(x):
        return T / (2 * x) - T / 4 * polygamma(1, x / 2)

    S_nu = 1.0
    nut = nu
    while abs(S_nu) > 10**(-5):                          # stopping criteria
        S_nu = df_nu(nut)
        H_nu = d2f_nu(nut)
        nut = nut - S_nu / H_nu
        if nut < 2:
            nut = 5
            H_nu = d2f_nu(nut)
            break
    Dnu = -1 / H_nu
    nuc = nut + np.sqrt(Dnu) * r.randn()
    if 2 < nuc < nu_ub:
        lalp_MH = f_nu(nuc) - f_nu(nu) - .5 * (nu - nut)**2 / Dnu + .5 * (nuc - nut)**2 / Dnu
        if np.exp(lalp_MH) > r.rand():
            nu = nuc
            flag = 1
    return nu, flag, f_nu
