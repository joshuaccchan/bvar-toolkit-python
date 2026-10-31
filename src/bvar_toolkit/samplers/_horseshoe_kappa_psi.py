import numpy as np

from .._rng import as_rng


def horseshoe_kappa_psi(theta, idx_kappa1, idx_kappa2, C, kappa, z_psi1, z_psi2, z_kappa,
                        rng=None):
    """bvt.samplers.horseshoe_kappa_psi - one sweep of the Minnesota-type HORSESHOE
    hierarchical shrinkage block for the coefficient vector theta (stacked k*n,
    intercept first per equation). kappa[0:2] are the global own-lag and
    other-lag scales; kappa[2:4] are returned unchanged.

      psi_kappa1, psi_kappa2, z_psi1, z_psi2, kappa, z_kappa = \\
          bvt.samplers.horseshoe_kappa_psi(theta, idx_kappa1, idx_kappa2, C,
                                           kappa, z_psi1, z_psi2, z_kappa)

    rng is None, an integer seed, a numpy.random.Generator or a bvt.Stream. The input kappa
    is not modified.

    Caller contract: the Psi reassembly, Psi[idx_kappa1] = psi_kappa1 and
    Psi[idx_kappa2] = psi_kappa2, stays with the caller.

    rng consumption, in order: rng.gamrnd n*p-vector (psi1), rng.gamrnd (n-1)*n*p-vector
    (psi2), rng.gamrnd n*p-vector (z_psi1), rng.gamrnd (n-1)*n*p-vector (z_psi2), two
    scalar rng.gamrnd (kappa1, kappa2), one 2-vector rng.gamrnd (z_kappa).

    NEVER merge with bvt.samplers.gig_shrinkage: that is the MAHP normal-gamma
    (GIG) block, a different prior family with a different draw sequence.

    See:
    Chan, J.C.C., Koop, G. and Yu, X. (2024). Large Order-Invariant Bayesian
    VARs with Stochastic Volatility, Journal of Business and Economic
    Statistics, 42(2): 825-837.
    """
    r = as_rng(rng)
    theta = np.asarray(theta, dtype=float).ravel()
    C = np.asarray(C, dtype=float).ravel()
    kappa = np.array(kappa, dtype=float).ravel()
    z_kappa = np.asarray(z_kappa, dtype=float).ravel()
    i1, i2 = np.asarray(idx_kappa1), np.asarray(idx_kappa2)
    np_ = i1.size                  # = n*p
    nnp = i2.size                  # = (n-1)*n*p

        # sample psi
    tmpv1 = 1 / z_psi1 + theta[i1]**2 / (2 * C[i1] * kappa[0])
    tmpv2 = 1 / z_psi2 + theta[i2]**2 / (2 * C[i2] * kappa[1])
    psi_kappa1 = 1 / r.gamrnd(1, 1 / tmpv1)
    psi_kappa2 = 1 / r.gamrnd(1, 1 / tmpv2)

        # sample z_psi
    z_psi1 = 1 / r.gamrnd(1, 1 / (1 + 1 / psi_kappa1))
    z_psi2 = 1 / r.gamrnd(1, 1 / (1 + 1 / psi_kappa2))

        # sample kappa1 and kappa2
    tmpc1 = 1 / z_kappa[0] + np.sum(theta[i1]**2 / (2 * psi_kappa1 * C[i1]))
    tmpc2 = 1 / z_kappa[1] + np.sum(theta[i2]**2 / (2 * psi_kappa2 * C[i2]))
    kappa[0] = 1 / r.gamrnd((np_ + 1) / 2, 1 / tmpc1)
    kappa[1] = 1 / r.gamrnd((nnp + 1) / 2, 1 / tmpc2)

        # sample z_kappa
    z_kappa = 1 / r.gamrnd(1, 1 / (1 + 1 / kappa[:2]))
    return psi_kappa1, psi_kappa2, z_psi1, z_psi2, kappa, z_kappa
