"""forecast_now - forecast the quarters after the end of the sample

The evaluation in build.py and your_data.py stops before the last observation, so that every
forecast has a realized value to be scored against. This script does the other thing: it
estimates through the final observation and reports the forecasts of the quarters that have
not happened yet.

For each model it prints the predictive mean, the 68 and 90 percent intervals and the
probability of an event the settings name - by default that quarterly GDP growth is negative.
Those come from the predictive distribution itself, which is an equally weighted mixture of
normals, one per posterior draw: bvt.forecast.simulate returns each draw's mean and standard
deviation, bvt.forecast.mixquantile inverts the mixture for the intervals, and the event
probability is the mixture cdf at the threshold. Fitting one normal to the mixture, or taking
quantiles of the draws' means, would both understate the uncertainty.

Nothing here is scored, because there is nothing yet to score it against.

Usage, from the root of the repository:  python tutorials/forecasting/forecast_now.py
"""

import os
import tempfile

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

TDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TDIR))

# ---- settings ----
file = os.path.join(REPO, 'tutorials', 'sv_specification', 'macro5_Q.csv')
cols = ['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']
vlabel = ['Unemployment', 'PCE inflation', 'Fed funds', 'NFCI', 'GDP growth']
datecol = 'Date'
p = 4                    # lags
n0 = 8                   # rows used as initial conditions
H = 4                    # quarters ahead
nsim = 5000              # draws kept per model
burnin = 1000            # draws discarded first
seed = 20260922
event = {'variable': 'GDPC1', 'below': 0}   # the event whose probability is reported
fanvar = 'GDPC1'         # the variable drawn as a fan chart
outdir = tempfile.gettempdir()   # where the report goes; '' for none

# ---- data ----
raw = pd.read_csv(file, float_precision='round_trip')
sel = raw[cols].to_numpy(dtype=float)
keep = np.all(np.isfinite(sel), axis=1)
lo, hi = np.flatnonzero(keep)[[0, -1]]
assert keep[lo:hi + 1].all(), 'the selected columns have missing values inside the sample'
data = sel[lo:hi + 1]
dates = pd.to_datetime(raw[datecol].iloc[lo:hi + 1]).reset_index(drop=True) if datecol else None
nobs, n = data.shape
k = 1 + n * p
Y0, Y = data[:n0], data[n0:]
cfg = {'ylag': data[-p:][::-1].T, 'H': H}       # no realized value: nothing to score
assert event['variable'] in cols and fanvar in cols, 'the event or fan variable is not among cols'
ievent, ifan = cols.index(event['variable']), cols.index(fanvar)


def quarter(d):
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


if dates is not None:
    fdates = [dates.iloc[-1] + pd.DateOffset(months=3 * j) for j in range(1, H + 1)]
    qlab = [quarter(d) for d in fdates]
    print(f'\nestimated through {quarter(dates.iloc[-1])}, forecasting {qlab[0]} to {qlab[-1]}')
else:
    qlab = [f'h = {j}' for j in range(1, H + 1)]
    print(f'\nestimated through observation {nobs}, forecasting {H} ahead')
print(f'{n} variables, {Y.shape[0]} observations after {n0} initial conditions, {nsim} draws after '
      f'{burnin} burn-in')

# ---- the three models, estimated through the last observation ----
mname = ['homoskedastic', 'VAR-CSV', 'VAR-OISV']
nm = len(mname)
MU = np.zeros((nsim, n, H, nm))      # each draw's conditional mean
SD = np.zeros((nsim, n, H, nm))      # and its standard deviation

    # homoskedastic, drawn directly from the natural conjugate posterior
_, X = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
T = Y.shape[0]
A0, VA0, nu0, S0, _ = bvt.priors.niw(p, [.2**2, 100], Y0, Y, 'mlvarsv_ncp')
KA = X.T @ X
KA[np.diag_indices(k)] += 1 / VA0
CKA = np.linalg.cholesky(KA)
Ahat = solve_triangular(CKA, solve_triangular(CKA, A0 / VA0[:, None] + X.T @ Y, lower=True),
                        lower=True, trans='T')
Shat = S0 + A0.T @ (A0 / VA0[:, None]) + Y.T @ Y - Ahat.T @ KA @ Ahat
Shat = (Shat + Shat.T) / 2
r = bvt.Stream(np.random.default_rng(seed))
for d in range(nsim):
    Sig = r.iwishrnd(Shat, nu0 + T)
    A = Ahat + solve_triangular(CKA, r.randn(k, n), lower=True, trans='T') \
        @ np.linalg.cholesky(Sig).T
    y1, _, _, s1 = bvt.forecast.simulate('gauss', {'A': A, 'Sig': Sig}, cfg)
    MU[d, :, :, 0], SD[d, :, :, 0] = y1.T, s1.T

r = bvt.Stream(np.random.default_rng(seed))
c = bvt.models.var_csv(Y0, Y, p, nsim=nsim, burnin=burnin, rng=r, draws=True).draws
for d in range(nsim):
    dr = {'A': c.A[d].reshape((k, n), order='F'), 'Sig': c.Sig[d].reshape((n, n), order='F'),
          'h_T': c.h[d, -1], 'phi': c.phi[d], 'sigh2': c.sigh2[d]}
    y1, _, _, s1 = bvt.forecast.simulate('csv', dr, cfg, rng=r)
    MU[d, :, :, 1], SD[d, :, :, 1] = y1.T, s1.T

r = bvt.Stream(np.random.default_rng(seed))
s = bvt.models.var_sv(Y0, Y, p, model='OI', nsim=nsim, burnin=burnin, rng=r, draws=True).draws
for d in range(nsim):
    dr = {'A': s.A[d].reshape((k, n), order='F'), 'impact': s.impact[d].reshape((n, n), order='F'),
          'h_T': s.h_T[d], 'phi': s.phi[d], 'sig2': s.sig2[d]}
    y1, _, _, s1 = bvt.forecast.simulate('oisv', dr, cfg, rng=r)
    MU[d, :, :, 2], SD[d, :, :, 2] = y1.T, s1.T

# ---- the predictive distribution, variable by variable ----
lv = [.05, .16, .50, .84, .95]
Q = np.zeros((n, H, nm, len(lv)))
pev = np.zeros((H, nm))
for im in range(nm):
    for j in range(H):
        for i in range(n):
            Q[i, j, im] = bvt.forecast.mixquantile(MU[:, i, j, im], SD[:, i, j, im], lv)
        pev[j, im] = np.mean(ndtr((event['below'] - MU[:, ievent, j, im]) / SD[:, ievent, j, im]))

for im in range(nm):
    print(f'\n{mname[im]}')
    print(f"{'variable':<22s}" + ''.join(f'{q:>24s}' for q in qlab))
    for i in range(n):
        print(f'{vlabel[i][:22]:<22s}' + ''.join(
            f'{Q[i, j, im, 2]:10.2f} [{Q[i, j, im, 0]:5.1f},{Q[i, j, im, 4]:5.1f}]'
            for j in range(H)))
print('\nposterior median with the 90% interval in brackets')

print(f"\nprobability that {event['variable']} is below {event['below']:g}")
print(f"{'':<16s}" + ''.join(f'{q:>12s}' for q in qlab))
for im in range(nm):
    print(f'{mname[im]:<16s}' + ''.join(f'{100 * v:11.0f}%' for v in pev[:, im]))
print('read from the predictive distribution itself, the mixture over draws;\n'
      'a single normal fitted to its mean and variance would give another number')

# ---- the fan chart ----
back = 12
hist = np.arange(nobs - back, nobs)
if dates is not None:
    xh = dates.iloc[hist].to_numpy()
    xf = np.array(fdates, dtype='datetime64[ns]')
else:
    xh, xf = hist + 1, nobs + np.arange(1, H + 1)
x = np.r_[xh[-1:], xf]
last = data[-1, ifan]
fig, ax = plt.subplots(figsize=(7.8, 3.4))
bvt.util.shaded_band(x, np.r_[last, Q[ifan, :, 1, 0]], np.r_[last, Q[ifan, :, 1, 4]], .88, ax=ax)
bvt.util.shaded_band(x, np.r_[last, Q[ifan, :, 1, 1]], np.r_[last, Q[ifan, :, 1, 3]], .75, ax=ax)
ax.plot(xh, data[hist, ifan], 'k', linewidth=1.2)
ax.plot(x, np.r_[last, Q[ifan, :, 1, 2]], 'k--', linewidth=1.2)
ax.axhline(0, color='k', linestyle=':', linewidth=.8)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title(f'{vlabel[ifan]}: VAR-CSV forecast with 68% and 90% intervals')
fig.tight_layout()
fig.savefig(os.path.join(TDIR, 'fig_forecast_now.png'), dpi=150)

# ---- the report ----
if outdir:
    T1 = pd.DataFrame([(mname[im], qlab[j], cols[i], MU[:, i, j, im].mean(),
                        *Q[i, j, im, [0, 1, 3, 4]])
                       for im in range(nm) for j in range(H) for i in range(n)],
                      columns=['model', 'quarter', 'variable', 'mean', 'p05', 'p16', 'p84', 'p95'])
    T2 = pd.DataFrame([(mname[im], qlab[j], f"{event['variable']} below {event['below']:g}",
                        pev[j, im]) for im in range(nm) for j in range(H)],
                      columns=['model', 'quarter', 'event', 'probability'])
    meta = dict(file=file, columns=cols, p=p, n0=n0, n=n, T=Y.shape[0], H=H, nsim=nsim,
                burnin=burnin, seed=seed,
                estimated_through=quarter(dates.iloc[-1]) if dates is not None else nobs,
                event_variable=event['variable'], event_below=event['below'])
    bvt.util.report('forecast_now', {'forecasts': T1, 'event': T2}, meta, outdir)

plt.show()
