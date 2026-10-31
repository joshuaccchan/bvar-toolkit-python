"""guide - the code of guides/coefficient_draw/README.md: the Minnesota prior for the VAR(2) of
three variables of guides/var_stacking, with Sig replaced by the residual variances of univariate
AR(4) regressions; the posterior mean and a draw of the coefficients from one Cholesky factor;
the posterior mean against least squares; then a prior mean of one on the first own lags. The
sections called "the code to adapt" contain everything a sampler uses. The sections called "the
checks" form dense matrices, which is practical only for a small example.

Saves fig_shrinkage.png next to this file.

Usage, from the root of the repository:  python guides/coefficient_draw/guide.py
"""

import os

# one BLAS thread, set before NumPy loads (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def rel(a, b):
    """the largest difference between a and b, relative to the largest entry of b"""
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def row(x, digits):
    """a vector in brackets, with the given number of significant digits"""
    return '[' + ' '.join(f'{v:.{digits}g}' for v in np.ravel(x)) + ']'


# ---- 1. the data: the VAR(2) of guides/var_stacking, with four presample rows ----
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

# ---- 2. the prior: the code to adapt ----
c1, c2, c3 = .2**2, .1**2, 10**2   # own lags, other lags, intercepts
beta0, V, s2, _ = bvt.priors.minn(p, c1, c2, c3, Y0, Y, 4)
Sighat = np.diag(s2)               # Sig replaced by the AR(4) residual variances

# ---- 2. the checks ----
Vmat = V.reshape((k, n), order='F')   # column i holds the prior variances of equation i
own = np.zeros((k, n), dtype=bool)
other = np.zeros((k, n), dtype=bool)
lag = np.zeros(k, dtype=int)
for r in range(1, k):
    lag[r] = -(-r // n)            # ceil(r/n), the lag that row r holds
    j = (r - 1) % n                # the variable whose lag row r holds
    own[r, j] = True
    other[r, np.arange(n) != j] = True
print(f'\nthe AR(4) residual variances s2: {row(s2, 4)}')
print('prior standard deviations, equation 1: intercept, lags 1 and 2 of variables 1-3')
print(np.array2string(np.sqrt(Vmat[:, 0]), precision=4, floatmode='fixed', max_line_width=100))

# ---- 3. the posterior mean and a draw: the code to adapt ----
iV = sp.diags_array(1 / V)         # the prior precision, diagonal
Adraw, Ahat = bvt.samplers.var_coef_joint(Y, Z, Sighat, np.ones(T), iV, rng=rng)

# ---- 3. the checks ----
X = bvt.util.surform2(Z, n)
y = Y.ravel()
W = sp.kron(sp.eye_array(T), np.linalg.inv(Sighat), format='csc')
K = np.diag(1 / V) + (X.T @ W @ X).toarray()
bhat = np.linalg.solve(K, X.T @ (W @ y))
gapM = rel(bvt.util.vec(Ahat), bhat)
print(f'\nthe posterior mean against dense algebra on the stacked form: {gapM:.1e} (relative)')
R = 10000
B = np.zeros((k * n, R))
for r in range(R):
    B[:, r] = bvt.util.vec(bvt.samplers.var_coef_joint(Y, Z, Sighat, np.ones(T), iV, rng=rng)[0])
Kinv = np.linalg.inv(K)
sd = np.sqrt(np.diag(Kinv))
dmean = np.max(np.abs(B.mean(axis=1) - bhat) / sd)
dsd = np.max(np.abs(B.std(axis=1, ddof=1) / sd - 1))
dcorr = np.max(np.abs(np.corrcoef(B) - Kinv / np.outer(sd, sd)))
print(f'{R} draws: mean against the posterior mean, largest gap {dmean:.2f} posterior sd')
print(f'{R} draws: sd against the posterior sd, largest relative gap {dsd:.3f}')
print(f'{R} draws: correlations against the posterior correlations, largest gap {dcorr:.3f}')
Aeq = np.zeros((k, n))             # one regression per equation, since Sighat is diagonal
for i in range(n):
    Ki = np.diag(1 / Vmat[:, i]) + Z.T @ Z / s2[i]
    Aeq[:, i] = np.linalg.solve(Ki, Z.T @ Y[:, i] / s2[i])
gapE = rel(Aeq, Ahat)
print(f'equation by equation against the joint posterior mean: {gapE:.1e} (relative)')
Ols = np.linalg.solve(Z.T @ Z, Z.T @ Y)
_, Aloose = bvt.samplers.var_coef_joint(Y, Z, Sighat, np.ones(T), 1e-10 * sp.eye_array(k * n),
                                        rng=rng)
gapL = rel(Aloose, Ols)
print(f'a nearly flat prior against least squares: {gapL:.1e} (relative)')
Wtrue = sp.kron(sp.eye_array(T), np.linalg.inv(Sig), format='csc')
bhat_true = np.linalg.solve(np.diag(1 / V) + (X.T @ Wtrue @ X).toarray(), X.T @ (Wtrue @ y))
_, Ahat_true = bvt.samplers.var_coef_joint(Y, Z, Sig, np.ones(T), iV, rng=rng)
gapT = rel(bvt.util.vec(Ahat_true), bhat_true)
print(f'with the true, full Sig, against dense algebra: {gapT:.1e} (relative)')
assert gapM < 1e-10 and gapE < 1e-10 and gapL < 1e-8 and gapT < 1e-10, \
    'the posterior mean disagrees with dense algebra'
assert dmean < .1 and dsd < .05 and dcorr < .1, 'the draws do not match the posterior'

# ---- 4. the posterior mean against least squares ----
print('\nsum of absolute values, posterior mean over least squares:')
groups = [(own & (lag[:, None] == 1), 'own lag 1'),
          (own & (lag[:, None] == 2), 'own lag 2'),
          (other & (lag[:, None] == 1), 'other lag 1'),
          (other & (lag[:, None] == 2), 'other lag 2'),
          (np.tile(np.arange(k)[:, None] == 0, (1, n)), 'intercepts')]
for m, name in groups:
    print(f'  {name:<12s} {np.sum(np.abs(Ahat[m])) / np.sum(np.abs(Ols[m])):.2f}')
print(f'first own lags: least squares {row(np.diag(Ols[1:n + 1]), 3)}, '
      f'posterior mean {row(np.diag(Ahat[1:n + 1]), 3)}')
_, Vfree, _, _ = bvt.priors.minn(p, c1, 100, c3, Y0, Y, 4)   # c2 = 100: other lags almost unshrunk
_, Afree = bvt.samplers.var_coef_joint(Y, Z, Sighat, np.ones(T), sp.diags_array(1 / Vfree),
                                       rng=rng)
Rz = np.corrcoef(Z[:, 1:], rowvar=False)
print(f'first own lags with c2 = 100: {row(np.diag(Afree[1:n + 1]), 3)}; '
      f'correlation of the first lags of y1 and y2: {Rz[0, 1]:.2f}')
print(f'estimation error, mean absolute: least squares {np.mean(np.abs(Ols[1:] - A[1:])):.3f}, '
      f'posterior mean {np.mean(np.abs(Ahat[1:] - A[1:])):.3f}')

# ---- 5. a prior mean of one on the first own lags: the adaptation ----
A0 = np.zeros((k, n))
A0[1:n + 1] = np.eye(n)            # one on each first own lag
G, Ghat = bvt.samplers.var_coef_joint(Y - Z @ A0, Z, Sighat, np.ones(T), iV, rng=rng)
Adraw0 = A0 + G
Ahat0 = A0 + Ghat
bhat0 = np.linalg.solve(K, bvt.util.vec(A0) / V + X.T @ (W @ y))
gap0 = rel(bvt.util.vec(Ahat0), bhat0)
print(f'\nprior mean of one on the first own lags, against dense algebra: {gap0:.1e} (relative)')
assert gap0 < 1e-10, 'the draw with a nonzero prior mean disagrees with dense algebra'
print(f'posterior means of the first own lags: {row(np.diag(Ahat[1:n + 1]), 3)} '
      f'(prior mean zero), {row(np.diag(Ahat0[1:n + 1]), 3)} (prior mean one)')

# ---- the figure ----
fig, ax = plt.subplots(figsize=(3.94, 4.1))
lims = [-.2, .7]
ax.plot(lims, lims, ':', color=(.5, .5, .5), linewidth=.8)
for (m, name), mk in zip(groups[:4], ['o', 's', '^', 'v']):
    ax.plot(Ols[m], Ahat[m], mk, linestyle='none', markersize=6, markerfacecolor='none',
            markeredgecolor='k', label=name)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks([-.2, 0, .2, .4, .6])
ax.set_yticks([-.2, 0, .2, .4, .6])
ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('least squares')
ax.set_ylabel('posterior mean')
ax.tick_params(labelsize=8)
fig.legend(loc='lower center', ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, .12, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_shrinkage.png'), dpi=150)
plt.close(fig)
