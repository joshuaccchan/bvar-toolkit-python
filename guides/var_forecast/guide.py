"""guide - the code of guides/var_forecast/README.md: forecasts of the next 20 quarters from a
VAR(4) estimated under the natural conjugate prior on 240 quarters simulated from the true model
of guides/var_simulation. Independent draws of the parameters from their posterior, one
simulated path per draw, and the forecasts read off the paths; the same forecasts from the true
parameters and from the posterior means of the parameters; the checks; then the probability of
a recession. Sections 2 to 4 and the function ncp_lml above them are the code to adapt.

Reads guides/data/macro3_Q.csv. Saves fig_fan.png and fig_widths.png next to this file.

Usage, from the root of the repository:  python guides/var_forecast/guide.py
"""

import os

# one BLAS thread, set before NumPy loads (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
names = ['GDP growth', 'PCE inflation', 'unemployment']


def ncp_lml(Y, Z, VA, nu0, S0):
    """the log marginal likelihood under vec(A) | Sig ~ N(0, Sig kron diag(VA)) and
    Sig ~ IW(nu0, S0), with the posterior quantities Ahat, KA and Shat"""
    T, n = Y.shape
    KA = np.diag(1 / VA) + Z.T @ Z
    Ahat = np.linalg.solve(KA, Z.T @ Y)
    E = Y - Z @ Ahat                                   # the residuals at the posterior mean
    # S0 + Y'Y - Ahat' KA Ahat, without the cancellation
    Shat = S0 + E.T @ E + Ahat.T @ (Ahat / VA[:, None])
    Shat = (Shat + Shat.T) / 2
    lml = (-n * T / 2 * np.log(np.pi) - n / 2 * (np.sum(np.log(VA)) + bvt.util.ldet(KA))
           + nu0 / 2 * bvt.util.ldet(S0) - (nu0 + T) / 2 * bvt.util.ldet(Shat)
           + bvt.util.mgammaln(n, (nu0 + T) / 2) - bvt.util.mgammaln(n, nu0 / 2))
    return lml, Ahat, KA, Shat


def stationary(B, S):
    """the means and standard deviations of the stationary distribution of the VAR with
    coefficients B, intercept first, and error covariance matrix S"""
    k, n = B.shape
    p = (k - 1) // n
    F = np.vstack([B[1:].T, np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])])
    mu = np.linalg.solve(np.eye(n * p) - F, np.concatenate([B[0], np.zeros(n * (p - 1))]))
    Q = np.zeros((n * p, n * p))
    Q[:n, :n] = S
    V = np.linalg.solve(np.eye((n * p)**2) - np.kron(F, F), bvt.util.vec(Q))
    V = V.reshape((n * p, n * p), order='F')
    return mu[:n], np.sqrt(np.diag(V[:n, :n]))


def fmt3(M):
    """the rows of M, median, lower and upper bound, as 'median (lower, upper)'"""
    return ''.join(f'{f"{m:.2f} ({lo:.2f}, {hi:.2f})":>26s}' for m, lo, hi in M)


def sig(x, d=3):
    """a vector with d significant digits, as '[a b c]'"""
    return '[' + ' '.join(f'{v:.{d}g}' for v in np.ravel(x)) + ']'


# ---- 1. the true model of guides/var_simulation, and the sample of 240 quarters simulated there ----
D = pd.read_csv(os.path.join(os.path.dirname(HERE), 'data', 'macro3_Q.csv'),
                float_precision='round_trip')
p = 4
Yd, Zd = bvt.util.build_lags(D.iloc[:, 1:4].to_numpy(), p)
Atrue = np.linalg.lstsq(Zd, Yd, rcond=None)[0]
Ud = Yd - Zd @ Atrue
Sigtrue = Ud.T @ Ud / (Yd.shape[0] - Zd.shape[1])
rng = np.random.default_rng(2)
data = bvt.util.simulate_var(Atrue, Sigtrue, 240, rng=rng)[0]  # GDP growth, inflation, unemployment

# ---- 2. the posterior draws: the code to adapt ----
Y, Z = bvt.util.build_lags(data, p)               # the first four quarters are initial values
T, n = Y.shape
k = Z.shape[1]
s2 = bvt.priors.resid_var_ar4(data[:4], Y)        # AR(4) residual variances
nu0 = n + 2
S0 = np.diag(s2)                                  # Sig ~ IW(nu0, S0)
c3 = 10**2                                        # the intercepts


def VA(c1):
    """the diagonal of V_A"""
    return np.concatenate([[c3], np.kron(1 / np.arange(1, p + 1)**2, c1 / s2)])


lc = minimize_scalar(lambda lc: -ncp_lml(Y, Z, VA(np.exp(lc)), nu0, S0)[0],
                     bounds=(np.log(1e-3), np.log(10)), method='bounded',
                     options={'xatol': 1e-4}).x
c1 = np.exp(lc)                                   # the shrinkage chosen by the marginal likelihood
_, Ahat, KA, Shat = ncp_lml(Y, Z, VA(c1), nu0, S0)
nuhat = nu0 + T
R = 10000
LK = np.linalg.cholesky(KA)                       # KA = LK @ LK.T
Adraw = np.zeros((k, n, R))
Sdraw = np.zeros((n, n, R))
rng = np.random.default_rng(3)
stream = bvt.Stream(rng)                          # the draws of rng, under MATLAB's names
for r in range(R):
    Sdraw[:, :, r] = stream.iwishrnd(Shat, nuhat)   # Sig ~ IW(nuhat, Shat)
    Adraw[:, :, r] = Ahat + solve_triangular(LK.T, rng.standard_normal((k, n))) \
        @ np.linalg.cholesky(Sdraw[:, :, r]).T    # A given Sig

# ---- 3. one path per draw: the code to adapt ----
H = 20
ylast = data[-p:]                                 # the last four quarters, the oldest first
paths = np.zeros((H, n, R))
for r in range(R):
    paths[:, :, r] = bvt.util.simulate_var(Adraw[:, :, r], Sdraw[:, :, r], H, ylast, rng=rng)[0]

# ---- 4. the forecasts: the code to adapt ----
# 5 x H x n: 90% and 68% bounds, median; 'hazen' is the definition of MATLAB's quantile
q = np.quantile(paths, [.05, .16, .5, .84, .95], axis=2, method='hazen')
fmean = paths.mean(axis=2)                        # the point forecast under squared error loss
pneg = np.mean(paths[0, 0] < 0)                   # P(GDP growth < 0) next quarter

# ---- 5. the same forecasts from the true parameters, and from the posterior means ----
Sigbar = Shat / (nuhat - n - 1)                   # the posterior mean of Sig
ptrue = np.zeros((H, n, R))
pplug = np.zeros((H, n, R))
rng = np.random.default_rng(4)
for r in range(R):
    ptrue[:, :, r] = bvt.util.simulate_var(Atrue, Sigtrue, H, ylast, rng=rng)[0]
    pplug[:, :, r] = bvt.util.simulate_var(Ahat, Sigbar, H, ylast, rng=rng)[0]
qtrue = np.quantile(ptrue, [.05, .16, .5, .84, .95], axis=2, method='hazen')
qplug = np.quantile(pplug, [.05, .16, .5, .84, .95], axis=2, method='hazen')

# ---- the results ----
print(f'\nVAR({p}) of {n} variables on {data.shape[0]} simulated quarters, T = {T}')
print('last four quarters of the sample:')
print(np.array2string(ylast, precision=4, floatmode='fixed'))
print(f'c1 chosen by the marginal likelihood: {c1:.4f} (sqrt {np.sqrt(c1):.3f}); '
      f'log ML {ncp_lml(Y, Z, VA(c1), nu0, S0)[0]:.2f}, '
      f'at c1 = 0.04 {ncp_lml(Y, Z, VA(.04), nu0, S0)[0]:.2f}')
Ols = np.linalg.lstsq(Z, Y, rcond=None)[0]
print(f'first own lags: true {sig(np.diag(Atrue[1:n + 1]))}, least squares '
      f'{sig(np.diag(Ols[1:n + 1]))}, posterior mean {sig(np.diag(Ahat[1:n + 1]))}')
mods = np.zeros(R)
low = np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])
for r in range(R):
    mods[r] = np.max(np.abs(np.linalg.eigvals(np.vstack([Adraw[1:, :, r].T, low]))))
qst = np.quantile(paths[:, :, mods < 1], [.05, .16, .5, .84, .95], axis=2, method='hazen')
print(f'draws with largest modulus above one: {np.sum(mods >= 1)} of {R}; without them, the '
      f'largest change in a quantile {np.max(np.abs(qst - q)):.3f}')
mtrue, sdtrue = stationary(Atrue, Sigtrue)
mhat, sdhat = stationary(Ahat, Sigbar)
print(f'means: true {sig(mtrue)}, at the posterior means {sig(mhat)}, '
      f'sample {sig(data.mean(axis=0))}')
print(f'sds: true {sig(sdtrue)}, at the posterior means {sig(sdhat)}, '
      f'sample {sig(data.std(axis=0, ddof=1))}')
Wb = q[4] - q[0]                                  # the widths of the 90% intervals
Wp = qplug[4] - qplug[0]
Wt = qtrue[4] - qtrue[0]
for h in [1, 4, 8, 20]:
    print(f'\nh = {h:2d}                 ' + ' '.join(f'{s:>26s}' for s in names))
    print(f'  BVAR    median, 90%:  {fmt3(q[[2, 0, 4], h - 1].T)}')
    print(f'  true    median, 90%:  {fmt3(qtrue[[2, 0, 4], h - 1].T)}')
    print(f'  plug-in median, 90%:  {fmt3(qplug[[2, 0, 4], h - 1].T)}')
    print('  90% widths, BVAR / plug-in / true: ' + ''.join(
        f'{Wb[h - 1, i]:7.2f} {Wp[h - 1, i]:6.2f} {Wt[h - 1, i]:6.2f}   ' for i in range(n)))
print(f'\nlargest gap between the mean and the median of the paths: '
      f'{np.max(np.abs(fmean - q[2])):.3f}')
print(f'largest gap between the BVAR and true medians over the {H} horizons: '
      f'{sig(np.max(np.abs(q[2] - qtrue[2]), axis=0))}')
print(f'BVAR median of GDP growth below the true one at every horizon: '
      f'{int(np.all(q[2, :, 0] < qtrue[2, :, 0]))}')
print(f'90% widths, BVAR over plug-in: h = 1 {sig(Wb[0] / Wp[0])}, '
      f'h = 20 {sig(Wb[H - 1] / Wp[H - 1])}')
print(f'90% widths, BVAR over true: h = 1 {sig(Wb[0] / Wt[0])}, '
      f'h = 20 {sig(Wb[H - 1] / Wt[H - 1])}')
print(f'90% widths, plug-in over true: h = 1 {sig(Wp[0] / Wt[0])}, '
      f'h = 20 {sig(Wp[H - 1] / Wt[H - 1])}')
print(f'probability of negative GDP growth next quarter: BVAR {pneg:.3f}, '
      f'true {np.mean(ptrue[0, 0] < 0):.3f}, plug-in {np.mean(pplug[0, 0] < 0):.3f}')
x = np.concatenate([[1], ylast[::-1].ravel()])    # x_{T+1} = (1, y_T', ..., y_{T-3}')
m1 = np.einsum('i,ijr->jr', x, Adraw)             # n x R: the one-step means, one per draw
print(f'one-step means: true {sig(x @ Atrue)}, posterior mean {sig(m1.mean(axis=1))}, '
      f'posterior sd {sig(m1.std(axis=1, ddof=1))}')

# ---- 6. the checks ----
# the one-step predictive density is multivariate t with nuhat - n + 1 degrees of freedom
df = nuhat - n + 1
loc = x @ Ahat
scl = np.sqrt(np.diag(Shat) * (1 + x @ np.linalg.solve(KA, x)) / df)
pr = np.array([.05, .5, .95])
qt = loc + scl * tdist.ppf(pr, df)[:, None]       # 3 x n
q1 = q[[0, 2, 4], 0]
psd = paths.std(axis=2, ddof=1)                   # H x n, the predictive sds
gapt = np.max(np.abs(q1 - qt) / psd[0])
print(f'\none-step quantiles against the t: largest gap {gapt:.3f} predictive sds')
# the quantiles at every horizon against the mixture of normals from the same draws
mu, sd = bvt.forecast.predictive(Adraw.reshape(k * n, R, order='F').T, Sdraw.transpose(2, 0, 1),
                                 ylast[::-1].T, H)
gapq = np.zeros((H, n, 3))
for h in range(H):
    for i in range(n):
        gapq[h, i] = (bvt.forecast.mixquantile(mu[:, i, h], sd[:, i, h], pr)
                      - q[[0, 2, 4], h, i]) / psd[h, i]
print(f'quantiles at every horizon against bvt.forecast.mixquantile: largest gap '
      f'{np.max(np.abs(gapq)):.3f} predictive sds')
# the draws: means against the posterior means of A and Sig
Asd = np.sqrt(np.outer(np.diag(np.linalg.inv(KA)), np.diag(Sigbar)))   # k x n posterior sds of A
print(f'mean of the draws of A against Ahat: largest gap '
      f'{np.max(np.abs(Adraw.mean(axis=2) - Ahat) / Asd):.3f} posterior sds')
print(f'mean of the draws of Sig against Shat/(nuhat - n - 1): largest relative gap '
      f'{np.max(np.abs(Sdraw.mean(axis=2) - Sigbar) / np.abs(Sigbar)):.4f}')
# the true model's paths against its exact predictive means and sds
mut, sdt = bvt.forecast.predictive(bvt.util.vec(Atrue)[None], Sigtrue[None], ylast[::-1].T, H)
mut, sdt = mut[0].T, sdt[0].T
print(f'true paths against the exact predictive: means '
      f'{np.max(np.abs(ptrue.mean(axis=2) - mut) / sdt):.3f} sds, sds '
      f'{np.max(np.abs(ptrue.std(axis=2, ddof=1) / sdt - 1)):.4f} (largest gaps)')
assert gapt < .1 and np.max(np.abs(gapq)) < .1, \
    'the simulated quantiles disagree with the exact ones'

# ---- 7. the probability of a recession: the adaptation ----
g = paths[:8, 0]                                  # GDP growth, 8 quarters by R paths
prec = np.mean(np.any((g[:7] < 0) & (g[1:] < 0), axis=0))   # two negative quarters in a row
gt, gp = ptrue[:8, 0], pplug[:8, 0]
print(f'\nprobability of a recession in the next eight quarters: BVAR {prec:.3f}, '
      f'true {np.mean(np.any((gt[:7] < 0) & (gt[1:] < 0), axis=0)):.3f}, '
      f'plug-in {np.mean(np.any((gp[:7] < 0) & (gp[1:] < 0), axis=0)):.3f}')
print(f'probability of negative growth in each of the eight quarters, BVAR: '
      f'{sig(np.mean(g < 0, axis=1), 2)}')

# ---- the figures ----
fig, axes = plt.subplots(1, n, figsize=(7.1, 3.15))
th = np.arange(201, 241)                          # the last 40 quarters, numbered from 1
tf = 240 + np.arange(H + 1)                       # the last quarter and the 20 that follow
for i, ax in enumerate(axes):
    h90 = bvt.util.shaded_band(tf, np.r_[data[-1, i], q[0, :, i]], np.r_[data[-1, i], q[4, :, i]],
                               .85, ax=ax)
    h68 = bvt.util.shaded_band(tf, np.r_[data[-1, i], q[1, :, i]], np.r_[data[-1, i], q[3, :, i]],
                               .7, ax=ax)
    hs, = ax.plot(th, data[th - 1, i], 'k', lw=.7)
    hm, = ax.plot(tf, np.r_[data[-1, i], q[2, :, i]], 'k', lw=1.4)
    ht, = ax.plot(tf, np.r_[data[-1, i], qtrue[2, :, i]], 'k--', lw=.8)
    ht90, = ax.plot(tf, np.r_[data[-1, i], qtrue[0, :, i]], 'k:', lw=.9)
    ax.plot(tf, np.r_[data[-1, i], qtrue[4, :, i]], 'k:', lw=.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(th[0], tf[-1])
    ax.set_title(names[i], fontsize=10)
    ax.tick_params(labelsize=8)
fig.legend([hs, hm, h68, h90, ht, ht90], ['Sample', 'BVAR median', 'BVAR 68% interval',
           'BVAR 90% interval', 'True model median', 'True model 90% interval'],
           loc='lower center', ncols=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .14, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_fan.png'), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, n, figsize=(7.1, 2.75))
hz = np.arange(1, H + 1)
for i, ax in enumerate(axes):
    hb, = ax.plot(hz, Wb[:, i], 'k', lw=1.4)
    hp, = ax.plot(hz, Wp[:, i], 'k-.', lw=.8)
    ht, = ax.plot(hz, Wt[:, i], 'k--', lw=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(1, H)
    ax.set_ylim(0, 1.05 * max(Wb[:, i].max(), Wp[:, i].max(), Wt[:, i].max()))
    ax.set_xlabel('quarters ahead', fontsize=9)
    ax.set_title(names[i], fontsize=10)
    ax.tick_params(labelsize=8)
fig.legend([hb, hp, ht], ['BVAR', 'Plug-in', 'True model'], loc='lower center', ncols=3,
           frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_widths.png'), dpi=150)
plt.close(fig)
