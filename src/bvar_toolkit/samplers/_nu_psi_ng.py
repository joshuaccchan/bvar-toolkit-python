import numpy as np
from scipy.special import digamma, gammaln, polygamma

from .._optim import fminbnd
from .._rng import as_rng


def nu_psi_ng(psi_kappa1, psi_kappa2, nu_psi, lam0_nu_psi, rng=None):
    """bvt.samplers.nu_psi_ng - independence-MH draw of the normal-gamma shape
    parameter nu_psi in the (Minnesota-type) normal-gamma priors of Chan (2021,
    IJF), given the local scales psi_kappa1/psi_kappa2 (psi ~ G(nu_psi, 2/nu_psi))
    under an Exp(lam0_nu_psi) prior: Newton-Raphson search for the conditional
    mode nu_psit (fminbnd fallback if an iterate goes negative), Gaussian
    N(nu_psit, -1/H) proposal, MH accept-reject.

    Trap: count is initialized to 0 and never incremented, so the count < 100
    guard on the while loop and the if count == 100 fminbnd fallback are dead
    code; the Newton iteration is bounded only by its convergence test and by the
    negative-iterate break. This is deliberate - leave the guard alone.

      nu_psi, flag, f_nu_psi = bvt.samplers.nu_psi_ng(psi_kappa1, psi_kappa2, nu_psi,
                                                     lam0_nu_psi)
      ... = bvt.samplers.nu_psi_ng(..., rng)

    Inputs:  psi_kappa1, psi_kappa2 - local scale draws (n*p and (n-1)*n*p)
             nu_psi                 - current draw
             lam0_nu_psi            - exponential prior rate
             rng                    - None, an integer seed, a numpy.random.Generator
                                      or a bvt.Stream
    Outputs: nu_psi, flag (1 if accepted), f_nu_psi (the log target kernel, a function)

    See:
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for
    Large Bayesian VARs, International Journal of Forecasting, 37(3), 1212-1226.
    """
    r = as_rng(rng)
    psi_kappa1 = np.asarray(psi_kappa1, dtype=float).ravel()
    psi_kappa2 = np.asarray(psi_kappa2, dtype=float).ravel()
    flag = 0
    n2p = psi_kappa1.size + psi_kappa2.size
    sum1 = np.sum(np.log(psi_kappa1)) + np.sum(np.log(psi_kappa2))
    sum2 = np.sum(psi_kappa1) + np.sum(psi_kappa2)

    def f_nu_psi(x):
        return n2p * (x * np.log(x / 2) - gammaln(x)) + (x - 1) * sum1 - x / 2 * sum2 \
            - lam0_nu_psi * x

    def df_nu_psi(x):
        return n2p * (np.log(x / 2) + 1 - digamma(x)) + sum1 - .5 * sum2 - lam0_nu_psi

    def d2f_nu_psi(x):
        return n2p / x - n2p * polygamma(1, x)

    S_nu_psi = 1.0
    nu_psit = nu_psi
    count = 0
    while abs(S_nu_psi) > 1e-5 and count < 100:          # stopping criteria
        S_nu_psi = df_nu_psi(nu_psit)
        H_nu_psi = d2f_nu_psi(nu_psit)
        nu_psit = nu_psit - S_nu_psi / H_nu_psi
        if nu_psit < 0:
            nu_psit = fminbnd(lambda x: -f_nu_psi(x), .01, 5)[0]
            H_nu_psi = d2f_nu_psi(nu_psit)
            break
    if count == 100:
        nu_psit = fminbnd(lambda x: -f_nu_psi(x), .01, 5)[0]
        H_nu_psi = d2f_nu_psi(nu_psit)

    Dnu_psi = -1 / H_nu_psi
    nu_psic = nu_psit + np.sqrt(Dnu_psi) * r.randn()
    if nu_psic > 0:
        alp_MH = f_nu_psi(nu_psic) - f_nu_psi(nu_psi) \
            - .5 * (nu_psi - nu_psit)**2 / Dnu_psi + .5 * (nu_psic - nu_psit)**2 / Dnu_psi
        if np.exp(alp_MH) > r.rand():
            nu_psi = nu_psic
            flag = 1
    return nu_psi, flag, f_nu_psi
