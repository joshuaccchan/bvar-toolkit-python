import numpy as np

from .._rng import as_rng


def gig_shrinkage(variant, beta, idx_kappa1, idx_kappa2, C, kappa, psi_kappa1, psi_kappa2,
                  nu_psi, c01, c02, n, p, psi_floor, rng=None):
    """bvt.samplers.gig_shrinkage - one sweep of the kappa/psi hierarchical
    shrinkage block of Chan (2021): the global shrinkage hyperparameters and, for
    the normal-gamma variants, the local scales psi_kappa1/psi_kappa2.

    The three variants are numerically DIFFERENT; never unify them:
      'mng'  -> kappa[0:2] with the Minnesota C, then the psi block;
      'ng'   -> a single kappa, no Minnesota C and no factor 2 - the NG prior
                variance is kappa*psi;
      'minn' -> kappa draws only, no psi block; psi_kappa1/psi_kappa2 are
                returned unchanged.

    rng consumption, all draws through rng.gigrnd, in order: 'mng' 2 + n*p +
    (n-1)*n*p calls; 'ng' 1 + n*p + (n-1)*n*p; 'minn' 2.

      kappa, psi_kappa1, psi_kappa2 = bvt.samplers.gig_shrinkage(variant, beta,
          idx_kappa1, idx_kappa2, C, kappa, psi_kappa1, psi_kappa2, nu_psi, c01, c02,
          n, p, psi_floor)
      ... = bvt.samplers.gig_shrinkage(..., rng)

    Inputs:  variant     - 'mng' | 'ng' | 'minn'
             beta        - n*(n*p+1) current coefficient draw
             idx_kappa1  - n*p own-lag positions (from bvt.priors.minnesota_C)
             idx_kappa2  - (n-1)*n*p other-lag positions
             C           - Minnesota second-moment vector ('mng'/'minn'; unused by 'ng')
             kappa       - current kappa: 4-vector, elements 0:2 updated ('mng'/'minn');
                           scalar ('ng')
             psi_kappa1  - n*p local scales ('mng'/'ng'; pass None for 'minn')
             psi_kappa2  - (n-1)*n*p local scales (idem)
             nu_psi      - normal-gamma shape ('mng'/'ng'; unused by 'minn')
             c01, c02    - gamma prior [shape, rate] pairs (c02 unused by 'ng')
             n, p        - VAR dimensions
             psi_floor   - psi lower bound against arithmetic underflow ('mng'/'ng';
                           unused by 'minn'). No default; the estimation samplers
                           use 1e-10, the MNG forecasting sampler 1e-16
             rng         - None, an integer seed, a numpy.random.Generator or a bvt.Stream
    Outputs: kappa, psi_kappa1, psi_kappa2 - updated state (psi unchanged under 'minn')

    The Psi reassembly, Psi[idx_kappa1] = psi_kappa1 and Psi[idx_kappa2] =
    psi_kappa2, stays with the caller. The inputs are not modified.

    See:
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for
    Large Bayesian VARs, International Journal of Forecasting, 37(3), 1212-1226.
    """
    r = as_rng(rng)
    beta = np.asarray(beta, dtype=float).ravel()
    idx_kappa1 = np.asarray(idx_kappa1).ravel()
    idx_kappa2 = np.asarray(idx_kappa2).ravel()
    if variant == 'mng':
        C = np.asarray(C, dtype=float).ravel()
        kappa = np.array(kappa, dtype=float).ravel()
        psi_kappa1 = np.array(psi_kappa1, dtype=float).ravel()
        psi_kappa2 = np.array(psi_kappa2, dtype=float).ravel()
            # sample kappa1 and kappa2
        tmpc1 = np.sum(beta[idx_kappa1]**2 / (2 * psi_kappa1 * C[idx_kappa1]))
        tmpc2 = np.sum(beta[idx_kappa2]**2 / (2 * psi_kappa2 * C[idx_kappa2]))
        kappa[0] = r.gigrnd(c01[0] - n * p / 2, 2 * c01[1], tmpc1)
        kappa[1] = r.gigrnd(c02[0] - (n - 1) * n * p / 2, 2 * c02[1], tmpc2)

            # sample psi
        tmpv1 = beta[idx_kappa1]**2 / (2 * C[idx_kappa1] * kappa[0])
        tmpv2 = beta[idx_kappa2]**2 / (2 * C[idx_kappa2] * kappa[1])
        for ik in range(n * p):            # lower bound psi_floor to avoid arithmetic underflow
            psi_kappa1[ik] = np.fmax(r.gigrnd(nu_psi - 1 / 2, nu_psi, tmpv1[ik]), psi_floor)
        for il in range((n - 1) * n * p):
            psi_kappa2[il] = np.fmax(r.gigrnd(nu_psi - 1 / 2, nu_psi, tmpv2[il]), psi_floor)
    elif variant == 'ng':
        psi_kappa1 = np.array(psi_kappa1, dtype=float).ravel()
        psi_kappa2 = np.array(psi_kappa2, dtype=float).ravel()
            # sample kappa
        tmpc1 = np.sum(beta[idx_kappa1]**2 / psi_kappa1)
        tmpc2 = np.sum(beta[idx_kappa2]**2 / psi_kappa2)
        kappa = r.gigrnd(c01[0] - n**2 * p / 2, 2 * c01[1], tmpc1 + tmpc2)

            # sample psi
        tmpv1 = beta[idx_kappa1]**2 / kappa
        tmpv2 = beta[idx_kappa2]**2 / kappa
        for ik in range(n * p):            # lower bound psi_floor to avoid arithmetic underflow
            psi_kappa1[ik] = np.fmax(r.gigrnd(nu_psi - 1 / 2, nu_psi, tmpv1[ik]), psi_floor)
        for il in range((n - 1) * n * p):
            psi_kappa2[il] = np.fmax(r.gigrnd(nu_psi - 1 / 2, nu_psi, tmpv2[il]), psi_floor)
    elif variant == 'minn':
        C = np.asarray(C, dtype=float).ravel()
        kappa = np.array(kappa, dtype=float).ravel()
            # sample kappa1 and kappa2
        tmpc1 = np.sum(beta[idx_kappa1]**2 / C[idx_kappa1])
        tmpc2 = np.sum(beta[idx_kappa2]**2 / C[idx_kappa2])
        kappa[0] = r.gigrnd(c01[0] - n * p / 2, 2 * c01[1], tmpc1)
        kappa[1] = r.gigrnd(c02[0] - (n - 1) * n * p / 2, 2 * c02[1], tmpc2)
    else:
        raise ValueError(f"unknown variant '{variant}'; use mng, ng or minn")
    return kappa, psi_kappa1, psi_kappa2
