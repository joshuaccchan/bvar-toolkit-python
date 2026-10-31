"""your_data - does modeling the volatility improve forecasts on your data?

Set the file, the columns, the lag length, the evaluation window and the chain lengths below,
then run the script. At every origin it re-estimates three reduced-form VARs on the sample to
that date - a homoskedastic one, one with a common volatility factor, and one with a volatility
process per equation under the order-invariant impact matrix - forecasts them to the longest
horizon in hs, and scores the forecasts by RMSFE and by the log predictive likelihood. It then
plots the running difference in log score and writes the comparison to csv files and a json
file. None of the three models depends on the order of the columns.

The defaults use the five series of the tutorial over a short window with short chains, which
takes about 80 seconds; the tutorial itself uses 140 origins and 5,000 draws after 1,000 burn-in.

Give cols as column names, which the first row of the file must hold. The selected columns
must have no missing values inside the sample; rows missing at either end are dropped. The
series must be stationary, since the prior means are zero.

Usage, from the root of the repository:  python tutorials/forecasting/your_data.py
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
from scipy.linalg import solve_triangular  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- settings ----
file = os.path.join(REPO, 'tutorials', 'sv_specification', 'macro5_Q.csv')
cols = ['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']
datecol = 'Date'         # the column holding the dates; '' to skip
p = 4                    # lags
n0 = 8                   # rows used as initial conditions
hs = [1, 4]              # horizons scored
nfirst = 24              # origins, counted back from the end of the sample
nsim = 1000              # draws kept per model and origin
burnin = 200             # draws discarded first
seed = 1
outdir = tempfile.gettempdir()   # where the report goes; '' for none

# ---- data ----
raw = pd.read_csv(file, float_precision='round_trip')
sel = raw[cols].to_numpy(dtype=float)
keep = np.all(np.isfinite(sel), axis=1)
assert keep.any(), 'every row of the selected columns has a missing value'
lo, hi = np.flatnonzero(keep)[[0, -1]]
assert keep[lo:hi + 1].all(), 'the selected columns have missing values inside the sample'
data = sel[lo:hi + 1]
flat = data.std(axis=0) == 0
assert not flat.any(), 'these series are constant: ' + ', '.join(np.array(cols)[flat])
assert n0 >= max(p, 4), 'n0 must be at least max(p,4)'
dates = pd.to_datetime(raw[datecol].iloc[lo:hi + 1]).to_numpy() if datecol else None
nobs, n = data.shape
k = 1 + n * p
H = max(hs)
origins = np.arange(nobs - nfirst, nobs)       # an origin t uses the first t rows
assert origins[0] > n0 + p + 10, 'not enough data before the first origin'
no = origins.size
odate = dates[origins - 1] if dates is not None else origins


def label(d):
    if dates is None:
        return f'row {d}'
    d = pd.Timestamp(d)
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


print(f'\n{n} variables, {no} origins ({label(odate[0])} to {label(odate[-1])}), {nsim} draws '
      f'after {burnin} burn-in')


def conjugate_draws(Y0, Y, r):
    """Independent draws from the natural conjugate posterior of the homoskedastic VAR."""
    T = Y.shape[0]
    _, X = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
    A0, VA0, nu0, S0, _ = bvt.priors.niw(p, [.2**2, 100], Y0, Y, 'mlvarsv_ncp')
    KA = X.T @ X
    KA[np.diag_indices(k)] += 1 / VA0
    CKA = np.linalg.cholesky(KA)
    Ahat = solve_triangular(CKA, solve_triangular(CKA, A0 / VA0[:, None] + X.T @ Y, lower=True),
                            lower=True, trans='T')
    Shat = S0 + A0.T @ (A0 / VA0[:, None]) + Y.T @ Y - Ahat.T @ KA @ Ahat
    Shat = (Shat + Shat.T) / 2
    for _ in range(nsim):
        Sig = r.iwishrnd(Shat, nu0 + T)
        A = Ahat + solve_triangular(CKA, r.randn(k, n), lower=True, trans='T') \
            @ np.linalg.cholesky(Sig).T
        yield {'A': A, 'Sig': Sig}


def keepscores(io, im, draws, spec, cfg, t, r):
    """Average the draws into the point forecast and the log predictive likelihood at the
    horizons that have a realized value."""
    yh, lj = [], []
    for dr in draws:
        y1, _, j1, _ = bvt.forecast.simulate(spec, dr, cfg, rng=r)
        yh.append(y1)
        lj.append(j1)
    yh, lj = np.array(yh), np.array(lj)
    for ih, h in enumerate(hs):
        if t + h <= nobs:
            point[io, :, ih, im] = yh[:, h - 1].mean(axis=0)
            ljnt[io, ih, im] = bvt.util.logsumexp(lj[:, h - 1]) - np.log(len(yh))


mname = ['homoskedastic', 'VAR-CSV', 'VAR-OISV']
nm = len(mname)
point = np.full((no, n, len(hs), nm), np.nan)
ljnt = np.full((no, len(hs), nm), np.nan)
actual = np.full((no, n, len(hs)), np.nan)

t0 = time.perf_counter()
for io, t in enumerate(origins):
    Y0, Y = data[:n0], data[n0:t]
    cfg = {'ylag': data[t - p:t][::-1].T, 'H': H, 'yobs': data[t:min(t + H, nobs)]}
    for ih, h in enumerate(hs):
        if t + h <= nobs:
            actual[io, :, ih] = data[t + h - 1]

        # homoskedastic, drawn directly from the natural conjugate posterior
    r = bvt.Stream(np.random.default_rng(seed + io + 1))
    keepscores(io, 0, conjugate_draws(Y0, Y, r), 'gauss', cfg, t, r)

        # common stochastic volatility
    r = bvt.Stream(np.random.default_rng(seed + io + 1))
    c = bvt.models.var_csv(Y0, Y, p, nsim=nsim, burnin=burnin, rng=r, draws=True).draws
    keepscores(io, 1, ({'A': c.A[d].reshape((k, n), order='F'),
                        'Sig': c.Sig[d].reshape((n, n), order='F'), 'h_T': c.h[d, -1],
                        'phi': c.phi[d], 'sigh2': c.sigh2[d]} for d in range(nsim)),
               'csv', cfg, t, r)

        # one volatility per equation, order invariant
    r = bvt.Stream(np.random.default_rng(seed + io + 1))
    s = bvt.models.var_sv(Y0, Y, p, model='OI', nsim=nsim, burnin=burnin, rng=r,
                          draws=True).draws
    keepscores(io, 2, ({'A': s.A[d].reshape((k, n), order='F'),
                        'impact': s.impact[d].reshape((n, n), order='F'), 'h_T': s.h_T[d],
                        'phi': s.phi[d], 'sig2': s.sig2[d]} for d in range(nsim)),
               'oisv', cfg, t, r)
print(f'{no} origins in {time.perf_counter() - t0:.0f} seconds')

# ---- the comparison ----
print(f'\naccuracy relative to the {mname[0]} VAR')
print(f"{'':<16s} {'horizon':>8s} {'RMSFE gain, percent':>20s} {'log score gain':>20s}")
gain = np.full((nm, len(hs)), np.nan)
lsg = np.full((nm, len(hs)), np.nan)
for im in range(1, nm):
    for ih in range(len(hs)):
        ok = ~np.isnan(actual[:, 0, ih])
        e0 = point[ok, :, ih, 0] - actual[ok, :, ih]
        em = point[ok, :, ih, im] - actual[ok, :, ih]
        gain[im, ih] = np.median(100 * (1 - np.sqrt(np.mean(em**2, axis=0))
                                        / np.sqrt(np.mean(e0**2, axis=0))))
        lsg[im, ih] = np.mean(ljnt[ok, ih, im] - ljnt[ok, ih, 0])
        print(f'{mname[im]:<16s} {hs[ih]:8d} {gain[im, ih]:20.1f} {lsg[im, ih]:20.3f}')
print('the RMSFE column is the median gain across the variables; the log score\n'
      'column is the mean gain in the joint log predictive likelihood per origin')
if no < 40:
    print(f'with only {no} origins these averages are noisy: raise nfirst before reading much in')

fig, ax = plt.subplots(figsize=(7.8, 3.8))
for im in range(1, nm):
    ok = ~np.isnan(ljnt[:, 0, im])
    ax.plot(odate[ok], np.cumsum(ljnt[ok, 0, im] - ljnt[ok, 0, 0]), linewidth=1.2,
            label=mname[im])
ax.axhline(0, color='k', linestyle=':', linewidth=.8)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='best', frameon=False)
ax.set_ylabel('cumulative log score difference')
ax.set_title(f'Density forecasts against the {mname[0]} VAR, {hs[0]} step ahead')
fig.tight_layout()

# ---- the report ----
if outdir:
    scores = pd.DataFrame([(mname[m], hs[ih], gain[m, ih], lsg[m, ih])
                           for m in range(nm) for ih in range(len(hs))],
                          columns=['model', 'horizon', 'rmsfe_gain_percent', 'log_score_gain'])
        # the forecasts, with the quarter each was made in and the quarter it is for, so that a
        # reader regrouping them cannot confuse the two
    rows = []
    for m in range(nm):
        for ih, h in enumerate(hs):
            for i in range(n):
                for io, t in enumerate(origins):
                    tgt = dates[t + h - 1] if dates is not None and t + h <= nobs else None
                    rows.append((mname[m], odate[io], tgt, cols[i], h, point[io, i, ih, m],
                                 actual[io, i, ih]))
    fc = pd.DataFrame(rows, columns=['model', 'origin_date', 'target_date', 'variable', 'horizon',
                                     'forecast', 'actual'])
    meta = dict(file=file, columns=cols, p=p, n0=n0, n=n, origins=no, first_origin=int(origins[0]),
                nsim=nsim, burnin=burnin, seed=seed, horizons=hs)
    bvt.util.report('forecasting_report', {'scores': scores, 'forecasts': fc}, meta, outdir)

plt.show()
