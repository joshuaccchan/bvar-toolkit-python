import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from .._struct import Struct
from ._kron_bvar import _grid, _grid_den, _lmeanexp, _niw_post, _s2
from ._lniwpdf import lniwpdf


def _nu_lden(T, sum1, sum2):
    """The log conditional density kernel of nu given the mixture scales."""
    def fnu(x):
        return T * (x / 2 * np.log(x / 2) - gammaln(x / 2)) - (x / 2 + 1) * sum1 - x / 2 * sum2
    return fnu


def kron_bvar_t(shortY, X, pri, est):
    """bvt.ml.kron_bvar_t - log marginal likelihood of the BVAR-t model of
    Chan (2020, JBES), Chib-style: analytic Student-t likelihood at the
    posterior means (A_mean, Sig_mean, nu_mean), NIW prior ordinate plus the
    flat nu prior 1/(nuub-2), and posterior ordinates Rao-Blackwellized over
    the stored lam draws - the (A,Sig) ordinate as a log-mean-exp of
    conditional NIW densities, the nu ordinate as the mean of grid-normalized
    conditional densities. Consumes no rng.

      ML, out = bvt.ml.kron_bvar_t(shortY, X, pri, est)

      pri: A0, VA0, nu0, S0, nuub
      est: nsims, store_A, store_Sig (running sums), store_nu, store_lam
      out: llike, lpri, lpost, store_lpost, A_mean, Sig_mean, nu_mean

    Known quirk, shared across this family: the nu grid normalization divides
    by nugrid[1]-nugrid[0] although inserting nu_mean leaves one interval
    unequal.

    See:
    Chan, J.C.C. (2020). Large Bayesian VARs: A flexible Kronecker error
    covariance structure, Journal of Business and Economic Statistics, 38(1), 68-79.
    """
    shortY = np.asarray(shortY, dtype=float)
    X = np.asarray(X, dtype=float)
    T, n = shortY.shape
    A0, VA0 = np.asarray(pri['A0'], dtype=float), np.asarray(pri['VA0'], dtype=float).ravel()
    nu0, S0, nuub = pri['nu0'], np.asarray(pri['S0'], dtype=float), pri['nuub']
    nsims = int(est['nsims'])
    store_lam = np.asarray(est['store_lam'], dtype=float).reshape(nsims, -1)

    A_mean = np.asarray(est['store_A'], dtype=float) / nsims
    Sig_mean = np.asarray(est['store_Sig'], dtype=float) / nsims
    nu_mean = float(np.mean(est['store_nu']))

        # evaluate the log likelihood
    CSig = np.linalg.cholesky(Sig_mean)
    s2 = _s2(shortY - X @ A_mean, CSig)
    llike = T * (gammaln((nu_mean + n) / 2) - gammaln(nu_mean / 2)
                 - n / 2 * np.log(nu_mean * np.pi)) \
        - T * np.sum(np.log(np.diag(CSig))) - (nu_mean + n) / 2 * np.sum(np.log(1 + s2 / nu_mean))
    lpri = lniwpdf(A_mean, Sig_mean, A0, sp.diags_array(1 / VA0), nu0, S0) + np.log(1 / (nuub - 2))

        # evaluate the posterior density
    store_lpost = np.zeros((nsims, 2))           # [log density of A Sig, density of nu]
    nugrid, nuidx = _grid(nu_mean, 2, nuub, 700)

    for isim in range(nsims):
        lam = store_lam[isim]

            # compute the conditional density of Sig and A
        Ahat, KA, Shat = _niw_post(X, shortY, 1 / lam, A0, VA0, S0)
        lden_ASig = lniwpdf(A_mean, Sig_mean, Ahat, KA, nu0 + T, Shat)

            # compute the conditional density of nu
        fnu = _nu_lden(T, np.sum(np.log(lam)), np.sum(1 / lam))
        den_nu = _grid_den(fnu(nugrid), nuidx, nugrid[1] - nugrid[0])

        store_lpost[isim] = [lden_ASig, den_nu]

    lpost = np.array([_lmeanexp(store_lpost[:, 0]), np.log(np.mean(store_lpost[:, 1]))])
    ML = llike + lpri - np.sum(lpost)
    return ML, Struct(llike=llike, lpri=lpri, lpost=lpost, store_lpost=store_lpost,
                      A_mean=A_mean, Sig_mean=Sig_mean, nu_mean=nu_mean)
