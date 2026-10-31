"""guide - the code of guides/var_simulation/README.md: a VAR(4) fitted by least squares to US real
GDP growth, PCE inflation and the unemployment rate, 1960Q1-2019Q4 (guides/data/macro3_Q.csv),
which is the true model of the series Forecasting with a VAR; its stability and stationary
distribution; a sample of 240 quarters simulated from it with bvt.util.simulate_var, and its
moments against those of the data; the lag lengths at which AIC and BIC are smallest; the
moments of 1,000 additional samples; the checks; then 20 quarters simulated from the data of
2019. Sections 1 to 3 are the code to adapt.

Saves fig_simulated.png next to this file.

Usage, from the root of the repository:  python guides/var_simulation/guide.py
"""

import os

# one BLAS thread, set before NumPy loads (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import block_diag  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
names = ['GDP growth', 'PCE inflation', 'unemployment']


def moments(X):
    """the mean, standard deviation and first-order autocorrelation of each column"""
    Xc = X - X.mean(axis=0)
    return np.vstack([X.mean(axis=0), X.std(axis=0, ddof=1),
                      np.sum(Xc[1:] * Xc[:-1], axis=0) / np.sum(Xc**2, axis=0)])


def corr_of(S, i, j):
    return S[i, j] / np.sqrt(S[i, i] * S[j, j])


def mat2str(x, d=3):
    """a vector printed as MATLAB's mat2str(x, d) prints it"""
    return '[' + ' '.join(f'{v:.{d}g}' for v in np.ravel(x)) + ']'


# ---- 1. the VAR(4) fitted to the data: the code to adapt ----
D = pd.read_csv(os.path.join(os.path.dirname(HERE), 'data', 'macro3_Q.csv'),
                float_precision='round_trip')
data = D.iloc[:, 1:4].to_numpy()          # GDP growth, PCE inflation, unemployment, 1960Q1-2019Q4
p = 4
Y, Z = bvt.util.build_lags(data, p)       # y_t from 1961Q1 on; x_t = (1, y_{t-1}', ..., y_{t-4}')'
T, n = Y.shape
k = Z.shape[1]
A = np.linalg.lstsq(Z, Y, rcond=None)[0]  # least squares, A = (Z'Z)^{-1} Z'Y
U = Y - Z @ A
Sig = U.T @ U / (T - k)                   # Sig = U'U/(T - k)

# ---- 2. stability and the stationary distribution: the code to adapt ----
F = np.vstack([A[1:].T,                                   # s_t = c + F s_{t-1} + e_t
               np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])])
rho = np.max(np.abs(np.linalg.eigvals(F)))                # the largest modulus; stable if below one
c = np.concatenate([A[0], np.zeros(n * (p - 1))])
mu = np.linalg.solve(np.eye(n * p) - F, c)                # mu = (I - F)^{-1} c
Q = block_diag(Sig, np.zeros((n * (p - 1), n * (p - 1))))
V = np.linalg.solve(np.eye((n * p)**2) - np.kron(F, F),   # vec(V) = (I - F kron F)^{-1} vec(Q)
                    bvt.util.vec(Q)).reshape(n * p, n * p, order='F')

# ---- 3. a simulated sample: the code to adapt ----
rng = np.random.default_rng(2)
Ysim, Y0sim = bvt.util.simulate_var(A, Sig, 240, rng=rng)   # 240 quarters, as many as in the data

# ---- the results ----
pop = np.vstack([mu[:n], np.sqrt(np.diag(V[:n, :n])),
                 np.diag(F[:n] @ V[:, :n]) / np.diag(V[:n, :n])])
print(f'\nVAR({p}) of {n} variables by least squares, T = {T}')
print(f'largest modulus of the companion eigenvalues: {rho:.3f}')
print(f'residual sd {mat2str(np.sqrt(np.diag(Sig)))}; correlations: '
      f'GDP-inflation {corr_of(Sig, 0, 1):.2f}, GDP-unemployment {corr_of(Sig, 0, 2):.2f}, '
      f'inflation-unemployment {corr_of(Sig, 1, 2):.2f}')
print(f'own lag-1 coefficients: {mat2str(np.diag(A[1:n + 1]))}')
print(f"\n{'':<14} {'mean':>22} {'sd':>22} {'autocorrelation':>22}")
print(f"{'':<14} {'data/sim/stationary':>22} {'data/sim/stationary':>22} "
      f"{'data/sim/stationary':>22}")
md = moments(data)
ms = moments(Ysim)
for i in range(n):
    print(f'{names[i]:<14} {md[0, i]:6.2f} {ms[0, i]:6.2f} {pop[0, i]:6.2f}    '
          f'{md[1, i]:6.2f} {ms[1, i]:6.2f} {pop[1, i]:6.2f}    '
          f'{md[2, i]:6.2f} {ms[2, i]:6.2f} {pop[2, i]:6.2f}')

isim = np.flatnonzero(Ysim[:, 2] == Ysim[:, 2].min())
print(f'lowest simulated unemployment rate: {Ysim[:, 2].min():.2f}, in quarter '
      f'{isim[0] + 1}')                       # quarters numbered from 1, as in the figure


def qname(i):
    return f'{1960 + i // 4}Q{i % 4 + 1}'


iu = np.flatnonzero(data[:, 2] == data[:, 2].min())
print(f'lowest unemployment rate in the data: {data[:, 2].min():.2f}, in '
      f"{' and '.join(qname(i) for i in iu)}")
irf = np.zeros(13)
S = np.zeros(n * p)
S[2] = 1                                  # unemployment's response to its own shock
for h in range(13):
    irf[h] = S[2]
    S = F @ S
hpk = np.argmax(irf)
print(f'response of unemployment to its own shock: peak {irf.max():.2f} after {hpk} quarters, '
      f'{irf[-1]:.2f} after 12')

# ---- the lag length: information criteria on a common sample ----
pmax = 8
crit = np.zeros((pmax, 2))
for q in range(1, pmax + 1):
    Yq, Zq = bvt.util.build_lags(data[pmax - q:], q)       # the same quarters for every q
    Uq = Yq - Zq @ np.linalg.lstsq(Zq, Yq, rcond=None)[0]
    Tq, kq = Yq.shape[0], Zq.shape[1]
    ld = np.log(np.linalg.det(Uq.T @ Uq / Tq))             # log|Sig_hat|, Sig_hat = U'U/T
    crit[q - 1] = [ld + 2 * n * kq / Tq, ld + np.log(Tq) * n * kq / Tq]   # AIC, BIC
best = np.argmin(crit, axis=0) + 1                         # lag lengths, which start at 1
print(f'\nlag length chosen on 1962Q1-2019Q4: AIC {best[0]}, BIC {best[1]}')

# ---- 4. 1,000 additional samples of 240 quarters ----
rng = np.random.default_rng(1)
nrep = 1000
mrep = np.zeros((3, n, nrep))
for r in range(nrep):
    mrep[:, :, r] = moments(bvt.util.simulate_var(A, Sig, 240, rng=rng)[0])
lo = np.quantile(mrep, .05, axis=2, method='hazen')
hi = np.quantile(mrep, .95, axis=2, method='hazen')
print(f'\n90% ranges over {nrep} simulated samples of 240 quarters:')
for i in range(n):
    print(f'{names[i]:<14} mean {lo[0, i]:5.2f} to {hi[0, i]:5.2f} (data {md[0, i]:5.2f}), '
          f'sd {lo[1, i]:4.2f} to {hi[1, i]:4.2f} (data {md[1, i]:4.2f}), '
          f'autocorrelation {lo[2, i]:4.2f} to {hi[2, i]:4.2f} (data {md[2, i]:4.2f})')
print(f'every moment of the data inside its 90% range: {int(np.all((md >= lo) & (md <= hi)))}')

# ---- 5. the checks ----
gapV = np.max(np.abs(V - F @ V @ F.T - Q)) / np.max(np.abs(V))
rng = np.random.default_rng(3)
Ylong = bvt.util.simulate_var(A, Sig, 200000, rng=rng)[0]
ml = moments(Ylong)
print(f"\nV against F V F' + Q: {gapV:.1e} (relative)")
print(f'200,000 simulated quarters against the stationary values: '
      f'means {np.max(np.abs(ml[0] - pop[0])):.3f}, sds {np.max(np.abs(ml[1] - pop[1])):.3f}, '
      f'autocorrelations {np.max(np.abs(ml[2] - pop[2])):.4f} (largest gaps)')
assert gapV < 1e-10, "the stationary covariance does not solve V = F V F' + Q"
assert np.max(np.abs(ml[0] - pop[0])) < .1 and np.max(np.abs(ml[1] / pop[1] - 1)) < .03, \
    'the long simulation does not match the stationary distribution'

# ---- 6. 20 quarters from the data of 2019: the adaptation ----
H, R = 20, 10000
paths = np.zeros((H, n, R))
rng = np.random.default_rng(4)
for r in range(R):
    paths[:, :, r] = bvt.util.simulate_var(A, Sig, H, data[-p:], rng=rng)[0]   # from 2019Q1-2019Q4
print(f'\n{R} paths from the data of 2019 (2019Q4: {mat2str(data[-1])}):')
for h in [4, 20]:
    qh = np.quantile(paths[h - 1], [.05, .5, .95], axis=1, method='hazen')   # h quarters ahead
    print(f'  h = {h:2d}: medians {mat2str(qh[1])}, 90% intervals {mat2str(qh[0])} to '
          f'{mat2str(qh[2])}')
print(f'  mean at h = 20 {mat2str(paths[H - 1].mean(axis=1))} against the stationary means '
      f'{mat2str(mu[:n])}')

# ---- the figure ----
fig, axes = plt.subplots(n, 2, figsize=(18 / 2.54, 13 / 2.54))
q = 1960 + np.arange(240) / 4
for i in range(n):
    yl = [min(data[:, i].min(), Ysim[:, i].min()), max(data[:, i].max(), Ysim[:, i].max())]
    ax = axes[i, 0]
    ax.plot(q, data[:, i], 'k', linewidth=.7)
    ax.set_xlim(q[0], q[-1])
    ax.set_ylim(yl)
    ax.set_title(f'{names[i]}, 1960-2019', fontsize=10)
    ax = axes[i, 1]
    ax.plot(np.arange(1, 241), Ysim[:, i], 'k', linewidth=.7)
    ax.set_xlim(1, 240)
    ax.set_ylim(yl)
    ax.set_title(f'{names[i]}, simulated', fontsize=10)
for ax in axes.ravel():
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_simulated.png'), dpi=150)
plt.close(fig)
