"""guide - the code of guides/natural_conjugate_ml/README.md: the VAR(2) of three variables of
guides/var_stacking under the natural conjugate prior, whose prior variances follow the
Minnesota prior with one hyperparameter c1 for all lag coefficients; the log marginal likelihood
in closed form, its maximum over c1, and the posterior means at the chosen value; the same
choice on 200 additional datasets; the checks against bvt.ml.kron_bvar; the choice of the lag
length; and the two hyperparameters of the asymmetric conjugate prior of tutorials/shrinkage.
Section 2 and the function ncp_lml above it are the code to adapt.

Saves fig_marglik.png next to this file.

Usage, from the root of the repository:  python guides/natural_conjugate_ml/guide.py
"""

import os

# one BLAS thread, set before NumPy loads (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def ncp_lml(Y, Z, VA, nu0, S0):
    """the log marginal likelihood under vec(A) | Sig ~ N(0, Sig kron diag(VA)) and
    Sig ~ IW(nu0, S0), with the posterior quantities Ahat, KA and Shat"""
    T, n = Y.shape
    KA = np.diag(1 / VA) + Z.T @ Z
    Ahat = np.linalg.solve(KA, Z.T @ Y)
    E = Y - Z @ Ahat                           # the residuals at the posterior mean
    # S0 + Y' Y - Ahat' KA Ahat, without the cancellation
    Shat = S0 + E.T @ E + Ahat.T @ (Ahat / VA[:, None])
    Shat = (Shat + Shat.T) / 2
    lml = (-n * T / 2 * np.log(np.pi) - n / 2 * (np.sum(np.log(VA)) + bvt.util.ldet(KA))
           + nu0 / 2 * bvt.util.ldet(S0) - (nu0 + T) / 2 * bvt.util.ldet(Shat)
           + bvt.util.mgammaln(n, (nu0 + T) / 2) - bvt.util.mgammaln(n, nu0 / 2))
    return lml, Ahat, KA, Shat


def simulate_var(a, A1, A2, C, burn, T, rng):
    """the data of section 1 from the current state of the generator, with the same presample
    rows"""
    n = a.size
    Yfull = np.zeros((burn + 2 + T, n))
    for t in range(2, burn + 2 + T):
        Yfull[t] = a + A1 @ Yfull[t - 1] + A2 @ Yfull[t - 2] + C @ rng.standard_normal(n)
    return Yfull[burn - 2:]


def mat2str(v, d):
    """the entries of v with d significant digits, in brackets, as MATLAB's mat2str"""
    return '[' + ' '.join(f'{x:.{d}g}' for x in v) + ']'


# ---- 1. the data of the earlier guides ----
n, p, T = 3, 2, 200
a = np.array([.5, .2, .1])                                  # intercepts
A1 = np.array([[.5, .1, 0], [.2, .4, .1], [0, .2, .5]])     # lag 1
A2 = np.array([[.2, 0, 0], [0, .1, 0], [.1, 0, .2]])        # lag 2
Sig = np.array([[1, .3, .2], [.3, .8, .1], [.2, .1, .6]])
A = np.vstack([a, A1.T, A2.T])     # k x n, A = (a, A_1, A_2)'
C = np.linalg.cholesky(Sig)
burn = 100
Yfull = np.zeros((burn + p + T, n))
for t in range(p, burn + p + T):
    Yfull[t] = a + A1 @ Yfull[t - 1] + A2 @ Yfull[t - 2] + C @ rng.standard_normal(n)
Yfull = Yfull[burn - 2:]           # y_{-3}, ..., y_0, y_1, ..., y_T
Y0 = Yfull[:4]                     # the presample of the AR(4) fits
Y, Z = bvt.util.build_lags(Yfull[2:], p)
k = Z.shape[1]
Atrue, Sigtrue = A, Sig
_, _, s2, _ = bvt.priors.minn(p, .2**2, .1**2, 10**2, Y0, Y, 4)   # the AR(4) residual variances

# ---- 2. the choice of c1: the code to adapt ----
nu0, S0 = n + 2, np.diag(s2)       # Sig ~ IW(nu0, S0)
c3 = 10**2                         # the intercepts


def VA(c1):
    """the diagonal of V_A"""
    return np.concatenate([[c3], np.kron(1 / np.arange(1, p + 1)**2, c1 / s2)])


def lml(c1):
    return ncp_lml(Y, Z, VA(c1), nu0, S0)


lc = minimize_scalar(lambda lc: -lml(np.exp(lc))[0], bounds=(np.log(1e-3), np.log(10)),
                     method='bounded').x   # maximize over log(c1)
c1hat = np.exp(lc)

# ---- 3. the marginal likelihood over a grid, and the posterior means ----
c1grid = np.logspace(-3, 1, 161)
lmlgrid = np.array([lml(c1)[0] for c1 in c1grid])
pathA = np.zeros((k, n, c1grid.size))
for j in range(c1grid.size):
    pathA[:, :, j] = lml(c1grid[j])[1]
Ols = np.linalg.solve(Z.T @ Z, Z.T @ Y)
lml_hat, Ahat = lml(c1hat)[:2]
lml_std, Astd = lml(.04)[:2]


def mae(B):
    """the mean absolute error over the 18 lag coefficients"""
    return np.mean(np.abs(B[1:] - Atrue[1:]))


jmax = int(np.argmax(lmlgrid))
print(f'\nthe chosen c1: {c1hat:.4f} (grid maximum at {c1grid[jmax]:.4f}), log ML {lml_hat:.2f}')
print(f'prior standard deviation of the first own lags, sqrt(c1): {np.sqrt(c1hat):.3f}')
print(f'log ML at c1 = 0.04: {lml_std:.2f}; at c1 = 0.001: {lml(1e-3)[0]:.2f}; '
      f'at c1 = 10: {lml(10)[0]:.2f}')
onepeak = bool(np.all(np.diff(lmlgrid[:jmax + 1]) > 0) and np.all(np.diff(lmlgrid[jmax:]) < 0))
print(f'on the grid, the log ML rises up to its maximum and falls after it: {int(onepeak)}')
print(f'at c1 = 10, lag coefficients against least squares, largest gap: '
      f'{np.max(np.abs(pathA[1:, :, -1] - Ols[1:])):.1e}')
assert onepeak and abs(np.log10(c1hat / c1grid[jmax])) <= 4 / 160, \
    'the maximum of minimize_scalar is not the single peak of the grid'
print(f'first own lags: least squares {mat2str(np.diag(Ols[1:n + 1]), 3)}, '
      f'chosen c1 {mat2str(np.diag(Ahat[1:n + 1]), 3)}, '
      f'c1 = 0.04 {mat2str(np.diag(Astd[1:n + 1]), 3)}')
print(f'estimation error, mean absolute: least squares {mae(Ols):.3f}, chosen c1 {mae(Ahat):.3f}, '
      f'c1 = 0.04 {mae(Astd):.3f}')

# ---- 4. the same choice on 200 additional datasets ----
rng = np.random.default_rng(2)
R = 200
mc = np.zeros((R, 4))   # chosen c1, then the error of least squares, chosen c1, c1 = 0.04
for r in range(R):
    Yr = simulate_var(a, A1, A2, C, burn, T, rng)
    Yr1, Zr = bvt.util.build_lags(Yr[2:], p)
    _, _, s2r, _ = bvt.priors.minn(p, .2**2, .1**2, 10**2, Yr[:4], Yr1, 4)

    def VAr(c1):
        return np.concatenate([[c3], np.kron(1 / np.arange(1, p + 1)**2, c1 / s2r)])

    lcr = minimize_scalar(lambda lc: -ncp_lml(Yr1, Zr, VAr(np.exp(lc)), nu0, np.diag(s2r))[0],
                          bounds=(np.log(1e-3), np.log(10)), method='bounded').x
    Bhat = ncp_lml(Yr1, Zr, VAr(np.exp(lcr)), nu0, np.diag(s2r))[1]
    Bstd = ncp_lml(Yr1, Zr, VAr(.04), nu0, np.diag(s2r))[1]
    mc[r] = [np.exp(lcr), mae(np.linalg.solve(Zr.T @ Zr, Zr.T @ Yr1)), mae(Bhat), mae(Bstd)]
print(f'\n{R} additional datasets: chosen c1 median {np.median(mc[:, 0]):.3f}, quartiles '
      f"{np.quantile(mc[:, 0], .25, method='hazen'):.3f} and "
      f"{np.quantile(mc[:, 0], .75, method='hazen'):.3f}")
err = mc[:, 1:].mean(axis=0)
print(f'  average error: least squares {err[0]:.3f}, chosen c1 {err[1]:.3f}, '
      f'c1 = 0.04 {err[2]:.3f}')
print(f'  chosen c1 more accurate than least squares in {np.sum(mc[:, 2] < mc[:, 1])}, '
      f'than c1 = 0.04 in {np.sum(mc[:, 2] < mc[:, 3])}')

# ---- 5. the checks: the identity of the marginal likelihood ----
gap = np.zeros(c1grid.size)
for j in range(c1grid.size):
    l, Aj, KA, Shat = ncp_lml(Y, Z, VA(c1grid[j]), nu0, S0)
    pri = bvt.Struct(A0=np.zeros((k, n)), VA0=VA(c1grid[j]), nu0=nu0, S0=S0)
    est = bvt.Struct(Ahat=Aj, Shat=Shat, KA=KA)
    gap[j] = l - bvt.ml.kron_bvar(Y, Z, pri, est)[0]
l, _, KA, Shat = ncp_lml(Y, Z, VA(c1hat), nu0, S0)
CS = np.linalg.cholesky(Sigtrue)
E = solve_triangular(CS, (Y - Z @ Atrue).T, lower=True).T   # (Y - Z Atrue) inv(CS')
llike = (-T * n / 2 * np.log(2 * np.pi) - T * np.sum(np.log(np.diag(CS)))
         - .5 * np.sum(E**2))      # log p(Y | A, Sig)
lpri = bvt.ml.lniwpdf(Atrue, Sigtrue, np.zeros((k, n)), sp.diags_array(1 / VA(c1hat)), nu0, S0)
lpost = bvt.ml.lniwpdf(Atrue, Sigtrue, Ahat, KA, nu0 + T, Shat)
print(f'\nclosed form against bvt.ml.kron_bvar, over the grid: largest gap '
      f'{np.max(np.abs(gap)):.1e}')
print(f'at the true values, log p(Y | A, Sig) + log p(A, Sig) - log p(A, Sig | Y) = '
      f'{llike + lpri - lpost:.6f}')
print(f'  against the closed form {l:.6f}: gap {llike + lpri - lpost - l:.1e}')
assert np.max(np.abs(gap)) < 1e-8 and abs(llike + lpri - lpost - l) < 1e-8, \
    'the closed form disagrees with the identity of the marginal likelihood'

# ---- 6. the lag length: the adaptation ----
pmax = 4
lmlp, c1p = np.zeros(pmax), np.zeros(pmax)
for q in range(1, pmax + 1):
    Yq, Zq = bvt.util.build_lags(Yfull[4 - q:], q)   # the same T rows of Y for every q
    assert np.array_equal(Yq, Y), 'the lag lengths explain different observations'

    def VAq(c1):
        return np.concatenate([[c3], np.kron(1 / np.arange(1, q + 1)**2, c1 / s2)])

    res = minimize_scalar(lambda lc: -ncp_lml(Yq, Zq, VAq(np.exp(lc)), nu0, S0)[0],
                          bounds=(np.log(1e-3), np.log(10)), method='bounded')
    c1p[q - 1], lmlp[q - 1] = np.exp(res.x), -res.fun
print(f'\nlag length: chosen c1 {mat2str(c1p, 3)}, log ML {mat2str(lmlp, 5)}')

# ---- 7. the asymmetric conjugate prior of tutorials/shrinkage: own and other lags apart ----
_, kappa = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, [.04, .01], 'redu')
_, kappa_b = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, [.2, .2], 'redu')   # another start
print(f'\nasymmetric conjugate prior: own lags {kappa[0]:.4f}, other lags {kappa[1]:.4f}')
print(f'  from another starting value: {kappa_b[0]:.4f}, {kappa_b[1]:.4f}')

# ---- the figure ----
fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.76))
axes[0].semilogx(c1grid, lmlgrid, 'k', lw=.8)
axes[0].set_title('Log marginal likelihood', fontsize=10)
own = np.zeros((k, n), dtype=bool)
own[1:n + 1] = np.eye(n, dtype=bool)
own[n + 1:2 * n + 1] = np.eye(n, dtype=bool)
lagrows = np.tile((np.arange(k) > 0)[:, None], (1, n))
paths = pathA.reshape(k * n, -1, order='F')
axes[1].semilogx(c1grid, paths[bvt.util.vec(lagrows & ~own)].T, color=(.6, .6, .6), lw=.8)
axes[1].semilogx(c1grid, paths[bvt.util.vec(own)].T, 'k', lw=.8)
axes[1].set_title('Posterior means', fontsize=10)
for ax in axes:
    ax.axvline(c1hat, color='k', ls='--', lw=.8, alpha=.7)
    ax.axvline(.04, color='k', ls=':', lw=.8, alpha=.7)
    ax.set_xlim(c1grid[0], c1grid[-1])
    ax.set_xlabel('$c_1$', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_marglik.png'), dpi=150)
plt.close(fig)
