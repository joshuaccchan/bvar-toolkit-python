"""guide - the code of guides/var_forecast_evaluation/README.md: the BVAR of guides/var_forecast and
an AR(4) for each variable, re-estimated at the end of each quarter from 120 to 239 of the sample
of guides/var_simulation on the data up to that quarter only, and forecasting one to four
quarters ahead; the forecasts scored against the outcomes by the root mean squared forecast
error, the log predictive likelihood and the coverage of the 90% intervals, together with the
forecasts of the true model; the checks. Section 2 and the functions ncp_forecast and ncp_lml at
the top are the code to adapt.

Reads guides/data/macro3_Q.csv. Saves fig_scores.png next to this file.

Usage, from the root of the repository:  python guides/var_forecast_evaluation/guide.py
"""

import os

# one BLAS thread, set before NumPy loads (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
names = ['GDP growth', 'PCE inflation', 'unemployment']


def ncp_forecast(Yt, yout, p, H, R, rng):
    """the forecasts of a VAR(p), or of an AR(p) when Yt has one column, under the natural
    conjugate prior with c1 chosen by the marginal likelihood, all estimated on Yt alone: the
    predictive means m, and the log predictive densities lp and the PITs of the outcomes yout
    (H x n, NaN where there is none), each n x H, from R draws of the parameters"""
    Y, Z = bvt.util.build_lags(Yt, p)
    T, n = Y.shape
    k = Z.shape[1]
    s2 = bvt.priors.resid_var_ar4(Yt[:4], Y)      # AR(4) residual variances, from Yt only

    def VA(c1):
        return np.concatenate([[10**2], np.kron(1 / np.arange(1, p + 1)**2, c1 / s2)])

    lc = minimize_scalar(lambda lc: -ncp_lml(Y, Z, VA(np.exp(lc)), n + 2, np.diag(s2))[0],
                         bounds=(np.log(1e-3), np.log(10)), method='bounded').x
    _, Ahat, KA, Shat = ncp_lml(Y, Z, VA(np.exp(lc)), n + 2, np.diag(s2))
    LK = np.linalg.cholesky(KA)
    r = bvt.Stream(rng)                           # the draws of rng, under MATLAB's names
    Ad = np.zeros((R, k * n))
    Sd = np.zeros((R, n, n))
    for d in range(R):
        S = r.iwishrnd(Shat, n + 2 + T)
        A = Ahat + solve_triangular(LK, r.randn(k, n), lower=True, trans='T') \
            @ np.linalg.cholesky(S).T
        Ad[d] = bvt.util.vec(A)
        Sd[d] = S
    mu, sd = bvt.forecast.predictive(Ad, Sd, np.flipud(Yt[-p:]).T, H)   # R x n x H
    z = (yout.T - mu) / sd
    m = mu.mean(axis=0)                           # the mean of the mixture over the draws
    lp = bvt.util.logsumexp(-np.log(sd) - z**2 / 2, 0) - np.log(R) - np.log(2 * np.pi) / 2
    pit = stats.norm.cdf(z).mean(axis=0)
    return m, lp, pit


def ncp_onestep(Yt, y1, p):
    """the log predictive density and the PIT of the outcome y1 (n values) one quarter ahead,
    from the Student-t predictive distribution of the same estimate as in ncp_forecast"""
    Y, Z = bvt.util.build_lags(Yt, p)
    T, n = Y.shape
    s2 = bvt.priors.resid_var_ar4(Yt[:4], Y)

    def VA(c1):
        return np.concatenate([[10**2], np.kron(1 / np.arange(1, p + 1)**2, c1 / s2)])

    lc = minimize_scalar(lambda lc: -ncp_lml(Y, Z, VA(np.exp(lc)), n + 2, np.diag(s2))[0],
                         bounds=(np.log(1e-3), np.log(10)), method='bounded').x
    _, Ahat, KA, Shat = ncp_lml(Y, Z, VA(np.exp(lc)), n + 2, np.diag(s2))
    x = np.concatenate([[1], np.flipud(Yt[-p:]).ravel()])
    df = T + 3                                    # nuhat - n + 1, with nuhat = n + 2 + T
    scl = np.sqrt(np.diag(Shat) * (1 + x @ np.linalg.solve(KA, x)) / df)
    z = (y1 - x @ Ahat) / scl
    lp = stats.t.logpdf(z, df) - np.log(scl)
    pit = stats.t.cdf(z, df)
    return lp, pit


def gauss_forecast(A, Sig, Yt, yout):
    """the same for the VAR with known coefficients A and error covariance matrix Sig"""
    k, n = A.shape
    p = (k - 1) // n
    H = yout.shape[0]
    mu, sd = bvt.forecast.predictive(bvt.util.vec(A)[None], Sig[None], np.flipud(Yt[-p:]).T, H)
    m, sd = mu[0], sd[0]
    z = (yout.T - m) / sd
    lp = -np.log(sd) - z**2 / 2 - np.log(2 * np.pi) / 2
    pit = stats.norm.cdf(z)
    return m, lp, pit


def ncp_lml(Y, Z, VA, nu0, S0):
    """the log marginal likelihood under vec(A) | Sig ~ N(0, Sig kron diag(VA)) and
    Sig ~ IW(nu0, S0), with the posterior quantities Ahat, KA and Shat"""
    T, n = Y.shape
    KA = np.diag(1 / VA) + Z.T @ Z
    Ahat = np.linalg.solve(KA, Z.T @ Y)
    E = Y - Z @ Ahat                              # the residuals at the posterior mean
    # S0 + Y'Y - Ahat' KA Ahat, without the cancellation
    Shat = S0 + E.T @ E + Ahat.T @ (Ahat / VA[:, None])
    Shat = (Shat + Shat.T) / 2
    lml = (-n * T / 2 * np.log(np.pi) - n / 2 * (np.sum(np.log(VA)) + bvt.util.ldet(KA))
           + nu0 / 2 * bvt.util.ldet(S0) - (nu0 + T) / 2 * bvt.util.ldet(Shat)
           + bvt.util.mgammaln(n, (nu0 + T) / 2) - bvt.util.mgammaln(n, nu0 / 2))
    return lml, Ahat, KA, Shat


def fmt(v):
    """a vector to three significant digits, in brackets"""
    return '[' + ' '.join(f'{x:.3g}' for x in v) + ']'


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
Tall, n = data.shape

# ---- 2. the recursive forecasts: the code to adapt ----
H, R = 4, 1000
origins = np.arange(120, Tall)                    # forecasts made at the end of quarters 120 to 239
no = origins.size
Yout = np.full((no, n, H), np.nan)                # the outcomes, NaN after the end of the sample
F = np.full((no, n, H, 3), np.nan)                # forecasts and scores: BVAR, AR(4), true model
LP, PIT = F.copy(), F.copy()
rng = np.random.default_rng(3)
for j in range(no):
    t = origins[j]
    Yt = data[:t]                                 # the data up to the origin, and nothing after it
    hmax = min(H, Tall - t)
    yout = np.full((H, n), np.nan)
    yout[:hmax] = data[t:t + hmax]
    Yout[j] = yout.T
    F[j, :, :, 0], LP[j, :, :, 0], PIT[j, :, :, 0] = ncp_forecast(Yt, yout, p, H, R, rng)
    for i in range(n):                            # an AR(4) for each variable
        F[j, i:i + 1, :, 1], LP[j, i:i + 1, :, 1], PIT[j, i:i + 1, :, 1] = \
            ncp_forecast(Yt[:, [i]], yout[:, [i]], p, H, R, rng)
    F[j, :, :, 2], LP[j, :, :, 2], PIT[j, :, :, 2] = gauss_forecast(Atrue, Sigtrue, Yt, yout)

# ---- 3. the scores ----
E = Yout[..., None] - F                           # forecast errors, no x n x H x 3
rmsfe = np.sqrt(np.nanmean(E**2, axis=0))         # n x H x 3
alpl = np.nanmean(LP, axis=0)                     # average log predictive likelihood
# the 90% coverage
cov90 = np.sum((PIT > .05) & (PIT < .95), axis=0) / np.sum(~np.isnan(PIT), axis=0)

# ---- the results ----
mods = ['BVAR', 'AR(4)', 'true']
print(f'\n{no} forecast origins, the ends of quarters {origins[0]} to {origins[-1]}')
for h in [1, 4]:
    print(f'\nh = {h}, {np.sum(~np.isnan(Yout[:, 0, h - 1]))} forecasts '
          f'{names[0]:>24s} {names[1]:>14s} {names[2]:>14s}')
    for m in range(3):
        print(f'  RMSFE, {mods[m]:<8s} {rmsfe[0, h - 1, m]:25.3f} {rmsfe[1, h - 1, m]:14.3f} '
              f'{rmsfe[2, h - 1, m]:14.3f}')
    for m in range(3):
        print(f'  average log score, {mods[m]:<8s} {alpl[0, h - 1, m]:13.3f} '
              f'{alpl[1, h - 1, m]:14.3f} {alpl[2, h - 1, m]:14.3f}')
    for m in range(3):
        print(f'  90% coverage, {mods[m]:<8s} {cov90[0, h - 1, m]:18.3f} '
              f'{cov90[1, h - 1, m]:14.3f} {cov90[2, h - 1, m]:14.3f}')
    print(f'  RMSFE over that of the AR(4): '
          f'BVAR {fmt(rmsfe[:, h - 1, 0] / rmsfe[:, h - 1, 1])}, '
          f'true {fmt(rmsfe[:, h - 1, 2] / rmsfe[:, h - 1, 1])}')
    print(f'  average log score minus that of the AR(4): '
          f'BVAR {fmt(alpl[:, h - 1, 0] - alpl[:, h - 1, 1])}, '
          f'true {fmt(alpl[:, h - 1, 2] - alpl[:, h - 1, 1])}')
print(f'\na correct 90% interval covers between {stats.binom.ppf(.025, no, .9) / no:.3f} and '
      f'{stats.binom.ppf(.975, no, .9) / no:.3f} of {no} independent outcomes, '
      f'with probability 0.95')
# cumulative log score gains of the BVAR, h = 1
c = np.cumsum(LP[:, :, 0, 0] - LP[:, :, 0, 1], axis=0)
print(f'cumulative log score gains of the BVAR over the AR(4), h = 1: to quarter 180 '
      f'{fmt(c[origins == 179][0])}, to quarter 240 {fmt(c[-1])}')

# ---- 4. the checks ----
# one quarter ahead, the predictive density is Student-t in closed form: against the mixture
gt = np.zeros((no, n, 2))                         # gaps in log densities and PITs: BVAR, AR(4)
gp = np.zeros((no, n, 2))
for j in range(no):
    Yt = data[:origins[j]]
    lpt, pitt = ncp_onestep(Yt, Yout[j, :, 0], p)
    gt[j, :, 0] = lpt - LP[j, :, 0, 0]
    gp[j, :, 0] = pitt - PIT[j, :, 0, 0]
    for i in range(n):
        lpt, pitt = ncp_onestep(Yt[:, [i]], Yout[j, [i], 0], p)
        gt[j, i, 1] = lpt[0] - LP[j, i, 0, 1]
        gp[j, i, 1] = pitt[0] - PIT[j, i, 0, 1]
gavg = np.max(np.abs(gt.mean(axis=0)))            # the gap in the average log scores
jw, iv, mw = np.unravel_index(np.argmax(np.abs(gt)), gt.shape)   # the largest gap
Yw = data[:origins[jw]]
yw = Yout[jw].T
kw = iv
if mw == 1:
    Yw, yw, kw = Yw[:, [iv]], yw[:, [iv]], 0
rng = np.random.default_rng(5)
_, lpw, _ = ncp_forecast(Yw, yw, p, H, 20000, rng)   # the same forecast from 20,000 draws
lpt = ncp_onestep(Yw, yw[0], p)[0]
print(f'\none quarter ahead, the closed form against the mixture of {R} draws: average log '
      f'scores {gavg:.4f}, single log densities {np.max(np.abs(gt)):.4f}, '
      f'PITs {np.max(np.abs(gp)):.4f} (largest gaps)')
print(f'  the largest gap, an outcome with PIT {PIT[jw, iv, 0, mw]:.4f}, is '
      f'{abs(lpt[kw] - lpw[kw, 0]):.4f} with 20,000 draws')
# the simulation error of the mixture four quarters ahead, at the first origin
rng = np.random.default_rng(6)
_, lp20, pit20 = ncp_forecast(data[:origins[0]], Yout[0].T, p, H, 20000, rng)
print(f'first origin, four quarters ahead, {R} draws against 20,000: log densities '
      f'{np.max(np.abs(lp20[:, H - 1] - LP[0, :, H - 1, 0])):.4f}, '
      f'PITs {np.max(np.abs(pit20[:, H - 1] - PIT[0, :, H - 1, 0])):.4f} (largest gaps)')
assert gavg < .005 and abs(lpt[kw] - lpw[kw, 0]) < .02, \
    'the mixture disagrees with the closed form'

# ---- 5. the PIT histogram: the adaptation ----
counts = np.zeros((n, 10), dtype=int)
for i in range(n):                                # BVAR, one quarter ahead, ten bins
    counts[i] = np.histogram(PIT[:, i, 0, 0], bins=np.linspace(0, 1, 11))[0]
print('\nPITs of the BVAR one quarter ahead, counts in ten bins:')
print(counts)
print(f'counts from {counts.min()} to {counts.max()}; one bin holds between '
      f'{stats.binom.ppf(.025, no, .1):.0f} and {stats.binom.ppf(.975, no, .1):.0f} of {no} '
      f'uniform PITs with probability 0.95')

# ---- the figure ----
fig, axes = plt.subplots(1, n, figsize=(7.09, 2.76))
for i, ax in enumerate(axes):
    hb, = ax.plot(origins + 1, np.cumsum(LP[:, i, 0, 0] - LP[:, i, 0, 1]), 'k', lw=1.2)
    ht, = ax.plot(origins + 1, np.cumsum(LP[:, i, 0, 2] - LP[:, i, 0, 1]), 'k--', lw=.7)
    ax.axhline(0, color='k', ls=':', lw=.7)
    ax.set_xlim(origins[0] + 1, Tall)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(names[i], fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
fig.legend([hb, ht], ['BVAR minus AR(4)', 'True model minus AR(4)'], loc='lower center',
           ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_scores.png'), dpi=150)
plt.close(fig)
