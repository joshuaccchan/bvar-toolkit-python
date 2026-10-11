import numpy as np

from .._rng import as_rng


def anormrnd(mu, rho, rng=None):
    """bvt.util.anormrnd - one draw from the two-component (bimodal) normal approximation to the
    absolute normal distribution of Villani (2009, Appendix C), used for the FIRST rotation
    coordinate zeta(1) in the row-wise B0 step of the order-invariant SVAR-SV sampler: mixture
    weight w = 1/(1+exp(2*mu/rho)) on the negative mode mu1 = mu/2 - sqrt(mu^2+4)/2 (else the
    positive mode mu2 = mu/2 + sqrt(mu^2+4)/2), each with variance muj^2*rho/(1+muj^2).

      draw = bvt.util.anormrnd(mu, rho)
      draw = bvt.util.anormrnd(mu, rho, rng)

      mu, rho : scalar parameters of the approximation above
      rng     : None, an integer seed, a numpy.random.Generator or a bvt.Stream
      draw    : scalar draw

    Not a truncated normal - do NOT merge into bvt.util.tnormrnd (different density, different
    rng sequence).

    See:
    Villani, M. (2009). Steady-State Priors for Vector Autoregressions, Journal of Applied
    Econometrics, 24(4): 630-650.
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian VARs with
    Stochastic Volatility, Journal of Business and Economic Statistics, 42(2): 825-837.
    """
    r = as_rng(rng)
    w = 1 / (1 + np.exp(2 * mu / rho))
    if w > r.rand():
        mu1 = mu / 2 - np.sqrt(mu ** 2 + 4) / 2
        sig21 = mu1 ** 2 * rho / (1 + mu1 ** 2)
        return mu1 + np.sqrt(sig21) * r.randn()
    mu2 = mu / 2 + np.sqrt(mu ** 2 + 4) / 2
    sig22 = mu2 ** 2 * rho / (1 + mu2 ** 2)
    return mu2 + np.sqrt(sig22) * r.randn()
