"""guide - the code of guides/gibbs_sampler/README.md: the VAR(2) of three variables of
guides/var_stacking under the independent normal and inverse-Wishart prior, with the Minnesota
variances of guides/coefficient_draw; a Gibbs sampler that draws the coefficients given Sig and
Sig given the coefficients; the checks of the chain by bvt.diag; the same loop under the natural
conjugate prior, whose posterior is known in closed form; then a forecast from the draws. The
section called "the code to adapt" contains the sampler.

Saves fig_chain.png next to this file.

Usage, from the root of the repository:  python guides/gibbs_sampler/guide.py
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


def gibbs_draws(Y, Z, iV, S0, nu0, nsim, burnin, rng):
    """the loop of Step 3, returning the draws of vec(A) and of the lower triangle of Sig, one
    row each"""
    T, n = Y.shape
    k = Z.shape[1]
    low = np.tril(np.ones((n, n), dtype=bool))
    Sig = S0
    draws = np.zeros((nsim, k * n + n * (n + 1) // 2))
    for isim in range(burnin + nsim):
        A = bvt.samplers.var_coef_joint(Y, Z, Sig, np.ones(T), iV, rng=rng)[0]
        U = Y - Z @ A
        Sig = bvt.Stream(rng).iwishrnd(S0 + U.T @ U, nu0 + T)
        if isim >= burnin:
            draws[isim - burnin] = np.concatenate([bvt.util.vec(A), Sig.T[low.T]])
    return draws


def row3(x):
    """a row vector to three significant digits, as MATLAB's mat2str(x, 3)"""
    return '[' + ' '.join(f'{v:.3g}' for v in x) + ']'


# ---- 1. the data and the prior of guides/coefficient_draw ----
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
_, V, s2, _ = bvt.priors.minn(p, .2**2, .1**2, 10**2, Y0, Y, 4)

# ---- 2. the Gibbs sampler: the code to adapt ----
nu0, S0 = n + 2, np.diag(s2)       # Sig ~ IW(nu0, S0), with prior mean S0
iV = sp.diags_array(1 / V)         # beta ~ N(0, V), the Minnesota variances
nsim, burnin = 20000, 1000
Sig = S0                           # the starting value
store_A, store_Sig = np.zeros((nsim, k, n)), np.zeros((nsim, n, n))
for isim in range(burnin + nsim):
    A = bvt.samplers.var_coef_joint(Y, Z, Sig, np.ones(T), iV, rng=rng)[0]   # beta given Sig
    U = Y - Z @ A
    Sig = bvt.Stream(rng).iwishrnd(S0 + U.T @ U, nu0 + T)                   # Sig given beta
    if isim >= burnin:
        store_A[isim - burnin] = A
        store_Sig[isim - burnin] = Sig

# ---- 3. the checks of the chain ----
low = np.tril(np.ones((n, n), dtype=bool))   # Sig.T[low.T]: the lower triangle, column by column
draws = np.hstack([store_A.transpose(0, 2, 1).reshape(nsim, k * n),
                   store_Sig.transpose(0, 2, 1)[:, low.T]])
truth = np.concatenate([bvt.util.vec(Atrue), Sigtrue.T[low.T]])
L = 100                            # Bartlett weights up to lag 100
IF = bvt.diag.inefficiency_factor(draws, L)
se = bvt.diag.mcse(draws, L)
Zg, pg, _ = bvt.diag.geweke(draws)
q = np.quantile(draws, [.05, .95], axis=0, method='hazen')
print(f'\n{nsim} draws after {burnin} burn-in, {draws.shape[1]} parameters')
print(f'inefficiency factors: coefficients {IF[:k * n].min():.2f} to {IF[:k * n].max():.2f}, '
      f'Sig {IF[k * n:].min():.2f} to {IF[k * n:].max():.2f}')
jmax = np.argmax(np.abs(Zg))
zmax = np.abs(Zg[jmax])
print(f'Geweke Z: largest |Z| {zmax:.2f}, for parameter {jmax} of 27; '
      f'{np.sum(np.abs(Zg) > 1.96)} of {Zg.size} beyond 1.96')
mz = np.zeros(10)                  # the same chain from ten other seeds
for s in range(10):
    rng = np.random.default_rng(s + 2)
    mz[s] = np.max(np.abs(bvt.diag.geweke(gibbs_draws(Y, Z, iV, S0, nu0, nsim, burnin, rng))[0]))
print(f'the largest |Z| over ten other seeds: {mz.min():.2f} to {mz.max():.2f}')
print(f'true values inside the 90% intervals: {np.sum((truth > q[0]) & (truth < q[1]))} '
      f'of {truth.size}')
names = ['A[1, 0], own lag 1 of eq 1', 'A[1, 1], lag 1 of y1 in eq 2', 'Sig[0, 0]', 'Sig[1, 0]']
cols = [1, k + 1, k * n, k * n + 1]
print(f"{'':30s} {'true':>6s} {'mean':>8s} {'90% interval':>18s} {'IF':>6s} {'MCSE':>8s}")
for name, c in zip(names, cols):
    print(f'{name:30s} {truth[c]:6.2f} {draws[:, c].mean():8.3f}   '
          f'({q[0, c]:6.3f}, {q[1, c]:6.3f}) {IF[c]:6.2f} {se[c]:8.4f}')

# ---- 4. the natural conjugate prior: the checks ----
VA = np.concatenate([[100], np.kron(1 / np.arange(1, p + 1)**2, .04 / s2)])   # prior variances
iVA = 1 / VA                       # of the rows of A, times Sig
Sig = S0
ncp_A, ncp_Sig = np.zeros((nsim, k, n)), np.zeros((nsim, n, n))
for isim in range(burnin + nsim):
    iVb = sp.kron(sp.csr_array(np.linalg.inv(Sig)), sp.diags_array(iVA))   # inv(Sig) kron inv(VA)
    A = bvt.samplers.var_coef_joint(Y, Z, Sig, np.ones(T), iVb, rng=rng)[0]
    U = Y - Z @ A
    Sig = bvt.Stream(rng).iwishrnd(S0 + U.T @ U + A.T @ (A * iVA[:, None]), nu0 + T + k)
    if isim >= burnin:
        ncp_A[isim - burnin] = A
        ncp_Sig[isim - burnin] = Sig
KA = np.diag(iVA) + Z.T @ Z        # the closed-form posterior
Ahat = np.linalg.solve(KA, Z.T @ Y)
Shat = S0 + Y.T @ Y - Ahat.T @ KA @ Ahat
ESig = Shat / (nu0 + T - n - 1)
sdA = np.sqrt(np.kron(np.diag(ESig), np.diag(np.linalg.inv(KA))))
ncp = np.hstack([ncp_A.transpose(0, 2, 1).reshape(nsim, k * n),
                 ncp_Sig.transpose(0, 2, 1)[:, low.T]])
exact_mean = np.concatenate([bvt.util.vec(Ahat), ESig.T[low.T]])
se_ncp = bvt.diag.mcse(ncp, L)
print('\nnatural conjugate prior, Gibbs against the closed form:')
zgap = np.max(np.abs(ncp.mean(axis=0) - exact_mean) / se_ncp)
sdgap = np.max(np.abs(ncp[:, :k * n].std(axis=0, ddof=1) / sdA - 1))
print(f'  posterior means: largest gap {zgap:.2f} Monte Carlo standard errors')
print(f'  posterior sd of the coefficients: largest relative gap {sdgap:.3f}')
assert zgap < 4.5 and sdgap < .05, 'the Gibbs sampler disagrees with the closed-form posterior'

# ---- 5. a forecast of y_{T+1} from the draws: the adaptation ----
xT1 = np.concatenate([[1], Y[-1], Y[-2]])   # x_{T+1}' = (1, y_T', y_{T-1}')
yT1 = np.zeros((nsim, n))
for r in range(nsim):
    yT1[r] = xT1 @ store_A[r] + np.linalg.cholesky(store_Sig[r]) @ rng.standard_normal(n)
qf = np.quantile(yT1, [.05, .95], axis=0, method='hazen')
condmean = xT1 @ store_A                     # x_{T+1}' A for each draw, nsim x n
dSig = np.diagonal(store_Sig, axis1=1, axis2=2)
print(f'\nthe forecast of y_{{T+1}}: mean {row3(yT1.mean(axis=0))}, '
      f'90% intervals {row3(qf[0])} to {row3(qf[1])}')
print("  mean against the average of the conditional means x'A: "
      f'{np.max(np.abs(yT1.mean(axis=0) - condmean.mean(axis=0))):.1e} (largest)')
vtot = dSig.mean(axis=0) + condmean.var(axis=0, ddof=1)   # mean(diag(Sig)) + var(x'A)
print("  variance against mean(diag(Sig)) + var(x'A): largest relative gap "
      f'{np.max(np.abs(yT1.var(axis=0, ddof=1) / vtot - 1)):.3f}')

# ---- the figure ----
fig, axes = plt.subplots(2, 4, figsize=(7.09, 3.15))
labels = [r'$A_{21}$: own lag 1, eq. 1', r'$A_{22}$: lag 1 of $y_1$, eq. 2', r'$\Sigma_{11}$',
          r'$\Sigma_{21}$']
for j, c in enumerate(cols):
    ax = axes[0, j]
    ax.plot(np.arange(1, nsim + 1), draws[:, c], 'k', linewidth=.5)
    ax.set_xlim(1, nsim)
    ax.set_xticks([5000, 10000, 15000, 20000])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(4, 4), useMathText=True)
    ax.set_title(labels[j], fontsize=9)
    ax = axes[1, j]
    ax.hist(draws[:, c], bins=50, density=True, color=(.75, .75, .75))
    ax.axvline(truth[c], color='k', linestyle='--', linewidth=.8)
for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.get_offset_text().set_fontsize(8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_chain.png'), dpi=150)
plt.close(fig)
