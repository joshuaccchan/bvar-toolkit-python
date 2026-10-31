"""your_data - compare stochastic volatility specifications of a VAR on your data

Set the file, the columns and the chain lengths below, then run the script. It estimates a
homoskedastic VAR and four VARs with stochastic volatility: common (VAR-CSV), Cholesky (VAR-SV,
under the asymmetric and the symmetric prior), factor (VAR-FSV, for each number of factors in
rs) and Cholesky with an outlier component (VAR-SVO). It then prints the log marginal
likelihood of each with its numerical standard error, plots the posterior probability that each
period is an outlier, and writes the comparison to csv files and a json file. The models,
priors and estimators are those of Chan (2023): four lags, the first eight rows as initial
conditions, and prior means of zero on the VAR coefficients, so the series must be stationary.
The defaults use the five series of the tutorial with short chains; the tutorial keeps 20,000
draws after 1,000 burn-in and uses 10,000 importance-sampling draws.

Give cols as column names, which the first row of the file must hold. The selected columns
must have no missing values inside the sample; rows missing at either end are dropped. With
rows = 'months' the file needs a date column in the book's convention (year + month/12), and
the months of each quarter are averaged; otherwise the rows are used as they are, and a date
column, when given, is checked for even spacing.

Usage, from the root of the repository:  python tutorials/sv_specification/your_data.py
"""

import os
import sys
import tempfile
import time

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

TDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TDIR)
import ml_varsv  # noqa: E402

# ---- settings ----
file = os.path.join(TDIR, 'macro5_Q.csv')
cols = ['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']    # the order is a normalization
pct = [False, False, False, False, False]   # which columns to multiply by 100
datecol = 'Date'                     # the column holding the dates; '' to skip the check
rows = 'periods'                     # 'months': average the months of each quarter;
                                     # 'periods': use the rows as they are
rs = [1, 2]                          # numbers of factors in VAR-FSV
nsim = 1000                          # draws kept
burnin = 200                         # draws discarded first
M = 1000                             # importance-sampling draws
seed = 1
outdir = tempfile.gettempdir()       # where the report goes; '' for none

# ---- data ----
raw = pd.read_csv(file, float_precision='round_trip')
sel = raw[cols].to_numpy(dtype=float)
sel[:, np.asarray(pct)] *= 100
keep = np.all(np.isfinite(sel), axis=1)
assert keep.any(), 'every row of the selected columns has a missing value'
lo, hi = np.flatnonzero(keep)[[0, -1]]
assert keep[lo:hi + 1].all(), 'the selected columns have missing values inside the sample'
data = sel[lo:hi + 1]
flat = data.std(axis=0) == 0
assert not flat.any(), 'these series are constant: ' + ', '.join(np.array(cols)[flat])
dates = raw[datecol].iloc[lo:hi + 1].to_numpy() if datecol else None
lab = [str(i + 1) for i in range(data.shape[0])]
if rows == 'months':
    assert dates is not None, 'averaging months into quarters needs a date column'
    am = np.round(12 * dates.astype(float)).astype(int)        # year*12 + month
    assert np.all(np.diff(am) == 1), 'the months kept are not consecutive'
    yr = (am - 1) // 12
    q, iq = np.unique(4 * yr + np.ceil((am - 12 * yr) / 3).astype(int), return_inverse=True)
    cnt = np.bincount(iq)
    avg = np.column_stack([np.bincount(iq, weights=data[:, j]) / cnt for j in range(len(cols))])
    data = avg[cnt == 3]
    q = q[cnt == 3]
    lab = [f'{(qq - 1) // 4}Q{qq - 4 * ((qq - 1) // 4)}' for qq in q]
elif dates is not None:
    try:
        d = pd.to_datetime(dates)
        step = np.diff(d.to_numpy()).astype('timedelta64[D]').astype(float)
        even = np.all(step > 0) and np.max(np.abs(step - np.median(step))) <= 3
        lab = [f'{x.year}Q{(x.month - 1) // 3 + 1}' for x in d]
    except (ValueError, TypeError):
        step = np.diff(dates.astype(float))
        even = np.all(step > 0) and \
            np.max(np.abs(step - np.median(step))) <= 1e-6 * max(1, abs(np.median(step)))
    assert even, 'the dates of the rows kept are not evenly spaced'
print(f'\n{len(cols)} variables, {data.shape[0] - 8} observations after 8 initial conditions '
      f'({lab[8]} to {lab[-1]})')

# ---- the specifications and their marginal likelihoods ----
spec = ([('VAR-NCP', False, 1), ('VAR-CSV', False, 1), ('VAR-SV', False, 1), ('VAR-SV', True, 1)]
        + [('VAR-FSV', False, r) for r in rs] + [('VAR-SVO', False, 1)])
label, lml, nse, secs, kap, hyper = [], [], [], [], [], []
for model, sym, r in spec:
    t0 = time.perf_counter()
    res = ml_varsv.run_ml(model, data, is_kappasym=sym, nsim=nsim, burnin=burnin, rng=seed, r=r,
                          M=M)
    lml.append(res.lml)
    nse.append(np.nan if res.lmlstd is None else res.lmlstd)
    lb = model + (', symmetric prior' if sym else '') + (f', r = {r}' if model == 'VAR-FSV' else '')
    label.append(lb)
    if model == 'VAR-SVO':
        pout = np.mean(res.store_o > 1, axis=0)
        omean = res.store_o.mean(axis=0)
    kap.append(np.ravel(res.kappa_mean).tolist() if 'kappa_mean' in res else None)
    hyper.append({k: v for k, v in res.Hyper.items()})
    secs.append(time.perf_counter() - t0)
    print(f'{lb}: {secs[-1]:.0f} s')
T, n, p = res.T, res.n, res.p          # the estimation sample, past the 8 initial conditions
lml, nse = np.array(lml), np.array(nse)

print('\nlog marginal likelihood (numerical standard error)')
best = int(np.argmax(lml))
for s in range(len(spec)):
    se = '    -' if np.isnan(nse[s]) else f'{nse[s]:5.2f}'
    print(f'{label[s]:<32s} {lml[s]:11.1f} ({se}) {lml[s] - lml[best]:9.1f}')
print(f"The last column is the difference from {label[best]}. VAR-NCP's marginal likelihood is\n"
      'available in closed form, so it has no standard error.')

print('\nperiods with a posterior outlier probability above 0.5 in VAR-SVO:'
      + ''.join(f' {lab[8 + t]}' for t in np.flatnonzero(pout > 0.5)))
fig, ax = plt.subplots(figsize=(7.8, 3.6))
ax.bar(np.arange(pout.size), pout, width=1, color=(.4, .4, .4), edgecolor='none')
tk = np.round(np.linspace(0, pout.size - 1, 6)).astype(int)
ax.set_xticks(tk, [lab[8 + t] for t in tk])
ax.set_ylim(0, 1)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('posterior probability')
ax.set_title('Probability that each period is an outlier, VAR-SVO')
fig.tight_layout()

# ---- the report ----
# One row per specification, the outlier probabilities of VAR-SVO in their own file, and the
# settings in the json file, which also holds the prior hyperparameters and the posterior mean
# of the shrinkage kappa of each model.
if outdir:
    models = pd.DataFrame({
        'model': label, 'symmetric_prior': [s[1] for s in spec],
        'r': [s[2] if s[0] == 'VAR-FSV' else np.nan for s in spec],    # factors, FSV only
        'log_ML': lml, 'nse': nse, 'gap_to_best': lml - lml[best], 'seconds': secs,
        'n': n, 'T': T, 'p': p, 'nsim': nsim, 'burnin': burnin, 'M': M, 'seed': seed})
    outliers = pd.DataFrame({'period': lab[8:], 'probability': pout, 'mean_multiplier': omean})
    meta = dict(file=file, columns=cols, pct=pct, rows=rows, datecol=datecol, n=n, T=T, p=p,
                rs=rs, nsim=nsim, burnin=burnin, M=M, seed=seed, sample=[lab[8], lab[-1]],
                kappa_mean=kap, hyper=hyper)
    bvt.util.report('sv_specification_report', {'models': models, 'outliers': outliers}, meta,
                    outdir)

plt.show()
