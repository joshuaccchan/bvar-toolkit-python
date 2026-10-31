"""your_data - a mixed-frequency VAR with common stochastic volatility on your data

Set the file, the columns, their names, which of them are quarterly, how each is transformed,
the sample and the chain length below, then run the script. It estimates the VAR of the
tutorial, with the priors of Chan, Poon and Zhu (2023), prints the monthly estimates of each
quarterly series over the last twelve months, plots the first quarterly series with its 68
percent credible band, and writes the monthly estimates to a csv and a json file. The defaults
use the tutorial's data from 1990 with short chains and take under a minute; the tutorial uses
20,000 draws after a burn-in period of 2,000.

The file has one row per month, with the dates in datecol in the format datefmt, such as
2026-07 for %Y-%m. A quarterly series holds its level in the last month of each quarter and
nothing in the other months. 'dlog' turns a series into 100 times its log change, quarter on
quarter for a quarterly series, so that its monthly values are monthly growth rates tied to it
by the aggregation of Mariano and Murasawa (2003); 'level' uses a monthly series as it is,
after multiplying by scale. A monthly series may have missing values anywhere, and when one in
levels is missing inside the sample, the growth rates around it keep their known sum. The
first p months serve as initial conditions.

Usage, from the root of the repository:  python tutorials/mixed_frequency/your_data.py
"""

import os
import tempfile
import time

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- settings ----
file = os.path.join(REPO, 'tutorials', 'mixed_frequency', 'mf_data.csv')
datecol = 'date'
datefmt = '%Y-%m'                              # format of the dates, first and last
cols = ['INDPRO', 'CPIAUCSL', 'UNRATE', 'PAYEMS', 'AWHMAN', 'GDPC1', 'FPIx']
names = ['IP', 'CPI', 'Unemployment', 'Payrolls', 'Hours', 'GDP', 'Investment']
quarterly = [False, False, False, False, False, True, True]
transform = ['dlog', 'dlog', 'level', 'dlog', 'level', 'dlog', 'dlog']
scale = [1, 1, 1, 1, .1, 1, 1]                 # multiplies a series used in levels
first = '1990-01'                              # first month of the sample; '' for all
last = ''                                      # last month; '' for all
p = 12                                         # lags
nsim = 1000                                    # draws kept
burnin = 500                                   # draws discarded first
seed = 1
outdir = tempfile.gettempdir()                 # where the report goes; '' for none

# ---- data ----
n = len(cols)
assert len(names) == n and len(quarterly) == n and len(transform) == n and len(scale) == n, \
    'cols, names, quarterly, transform and scale must have the same length'
assert all(t in ('dlog', 'level') for t in transform), "transform must be 'dlog' or 'level'"
quarterly = np.array(quarterly, dtype=bool)
transform = np.array(transform)
scale = np.array(scale, dtype=float)
assert not np.any(quarterly & (transform != 'dlog')), \
    "a quarterly series must be in levels, with transform 'dlog'"
tbl = pd.read_csv(file, float_precision='round_trip')
dates = pd.to_datetime(tbl[datecol], format=datefmt).to_numpy().astype('datetime64[M]')
assert np.all(np.diff(dates.astype(int)) == 1), 'the file must have one row per month, in order'
Lv = tbl[cols].to_numpy(dtype=float)
dl = Lv[:, transform == 'dlog']
assert np.all((dl > 0) | np.isnan(dl)), 'a series in log changes must be positive'


def month(s):
    return pd.to_datetime(s, format=datefmt).to_datetime64().astype('datetime64[M]')


# the window, with one month before it for the first growth rate
lo, hi = 1, dates.size - 1
if first:
    lo = int(np.flatnonzero(dates >= month(first))[0])
if last:
    hi = int(np.flatnonzero(dates <= month(last))[-1])
assert lo >= 1 and hi - lo > p + 12, 'the sample is too short or out of the file'
d = dates[lo:hi + 1]
T = d.size

X = np.full((T, n), np.nan)
# quarterly series: 100 x log change between consecutive quarterly levels in the file
for j in np.flatnonzero(quarterly):
    iobs = np.flatnonzero(~np.isnan(Lv[:, j]))
    assert np.all(dates[iobs].astype(int) % 12 % 3 == 2), \
        f'series {names[j]} must hold its values in the last month of each quarter'
    assert np.all(np.diff(iobs) == 3), f'series {names[j]} has a quarter missing inside its sample'
    g = np.full(Lv.shape[0], np.nan)
    g[iobs[1:]] = 100 * np.diff(np.log(Lv[iobs, j]))
    X[:, j] = g[lo:hi + 1]
# monthly series in log changes, with a restriction across each gap
jd = np.flatnonzero(~quarterly & (transform == 'dlog'))
G, Mg, zg = bvt.util.dlog_gaps(Lv[lo - 1:hi + 1][:, jd])
X[:, jd] = G
jl = np.flatnonzero(~quarterly & (transform == 'level'))
X[:, jl] = Lv[lo:hi + 1][:, jl] * scale[jl]
Mq, zq, Y, _ = bvt.util.mm_constraint(X, quarterly)
Mg = Mg.tocoo()
Mg = sp.csc_array((np.ones(Mg.nnz), (Mg.row, (Mg.col // jd.size) * n + jd[Mg.col % jd.size])),
                  shape=(Mg.shape[0], T * n))
M = sp.vstack([Mq, Mg], format='csc')
z = np.concatenate([zq, zg])

# the Minnesota scale of a quarterly series: 9/19 of the AR(4) residual variance of its
# quarterly growth rates, the monthly variance that gives it under the aggregation
sig2 = np.full(n, np.nan)
for j in np.flatnonzero(quarterly):
    g = X[~np.isnan(X[:, j]), j]
    assert g.size >= 12, f'series {names[j]} has fewer than 12 quarterly values in the sample'
    Zq = np.column_stack([np.ones(g.size - 4), g[3:-1], g[2:-2], g[1:-3], g[:-4]])
    e = g[4:] - Zq @ np.linalg.lstsq(Zq, g[4:], rcond=None)[0]
    sig2[j] = 9 / 19 * np.mean(e**2)
print(f'\n{d[0]} to {d[-1]}, {T} months, the first {p} as initial conditions; '
      f'{z.size} restrictions')
print('missing values: ' + ', '.join(f'{v} {k}' for v, k in zip(names, np.isnan(Y).sum(axis=0))))

# ---- estimation ----
t0 = time.perf_counter()
res = bvt.models.mfvar_csv(Y, p, M=M, z=z, sig2=sig2, nsim=nsim, burnin=burnin, rng=seed)
print(f'{nsim} draws after {burnin} burn-in: {time.perf_counter() - t0:.0f} s; share of h '
      f'blocks accepted {res.accept_rate:.2f}')

# ---- the monthly estimates of the quarterly series ----
jq = np.flatnonzero(quarterly)
for j in jq:
    print(f'\n{names[j]}, monthly growth in percent: posterior mean (90% band)')
    for t in range(T - 12, T):
        print(f'  {d[t]} {res.Y_mean[t, j]:7.2f} ({res.Y_q[t, j, 0]:6.2f}, '
              f'{res.Y_q[t, j, -1]:6.2f})')

j = jq[0]
k = d >= d[-1] - np.timedelta64(120, 'M')
dd = d.astype('datetime64[D]')
fig, ax = plt.subplots(figsize=(7.8, 3.6))
bvt.util.shaded_band(dd[k], res.Y_q[k, j, 1], res.Y_q[k, j, 3], ax=ax)
ax.plot(dd[k], res.Y_mean[k, j], 'k', linewidth=1.2)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, linewidth=.5, alpha=.5)
ax.set_title(f'{names[j]}, monthly growth: posterior mean and 68% credible band')
ax.set_ylabel('percent')
fig.tight_layout()

# ---- the report ----
# One row per month and quarterly series, and the settings in the json file.
if outdir:
    monthly = pd.DataFrame({
        'series': np.repeat([names[j] for j in jq], T),
        'month': np.tile(d.astype(str), jq.size),
        'mean': res.Y_mean[:, jq].T.ravel()})
    for i, q in enumerate(res.quantiles):
        monthly[f'q{round(100 * q):02d}'] = res.Y_q[:, jq, i].T.ravel()
    meta = dict(file=file, columns=cols, names=names, quarterly=quarterly, transform=transform,
                scale=scale, first=str(d[0]), last=str(d[-1]), p=p, nsim=nsim, burnin=burnin,
                seed=seed, sig2=res.sig2, accept_rate=res.accept_rate)
    bvt.util.report('mixed_frequency_report', monthly, meta, outdir)

plt.show()
