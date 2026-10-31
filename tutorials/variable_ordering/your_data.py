"""your_data - the Cholesky and order-invariant VARs with stochastic volatility on your data

Set the file, the columns, their names, the lag length and the chain length below, then run
the script. It estimates both models with the variables in the order given and in the reverse
order, prints how much each model's correlations change when the order is reversed, and plots
the correlation that changes most under the Cholesky model. The defaults use the four FRED-MD
series of the tutorial with short chains; Chan, Koop and Yu (2024) use 30,000 draws after
5,000 burn-in.

The file is a csv file and may have a header row; the column positions count every column of
the file from 0. The selected columns must be stationary, transformed as needed, and have no
missing values inside the sample; rows missing at either end are dropped. Setting datecol
checks that the rows kept are evenly spaced. The first n0 rows serve as initial conditions: at
least max(p, 4) of them.

Usage, from the root of the repository:  python tutorials/variable_ordering/your_data.py
"""

import itertools
import os
import time

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- settings ----
file = os.path.join(REPO, 'data', 'chan_koop_yu2024_jbes_oisv', 'FRED_MD_20vars.csv')
cols = [3, 5, 11, 12]                          # columns of the file, in the order to use
names = ['IP', 'unemployment', 'PCE inflation', 'fed funds']
datecol = None                                 # the column holding the dates; None to skip
p = 13                                         # lags
n0 = 24                                        # rows used as initial conditions
nsim = 1000                                    # draws kept
burnin = 200                                   # draws discarded first
seed = 1

# ---- data ----
raw = pd.read_csv(file, header=None, float_precision='round_trip')
raw = raw.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)   # a header row is NaN
sel = raw[:, cols]
assert len(names) == len(cols), 'names and cols must have the same length'
keep = np.all(np.isfinite(sel), axis=1)
assert keep.any(), 'every row of the selected columns has a missing value'
lo, hi = np.flatnonzero(keep)[[0, -1]]
assert keep[lo:hi + 1].all(), 'the selected columns have missing values inside the sample'
data = sel[lo:hi + 1]
flat = data.std(axis=0) == 0
assert not flat.any(), 'these series are constant: ' + ', '.join(np.array(names)[flat])
if datecol is not None:
    step = np.diff(raw[lo:hi + 1, datecol])
    assert np.all(step > 0) and np.max(np.abs(step - np.median(step))) <= \
        1e-6 * max(1, abs(np.median(step))), 'the dates of the rows kept are not evenly spaced'

# ---- estimation: both models, the order given and its reverse ----
Y0, Y = data[:n0], data[n0:]
n = len(cols)
rev = np.arange(n)[::-1]
runs = [('CS', False), ('CS', True), ('OI', False), ('OI', True)]
res = []
for model, reverse in runs:
    order = rev if reverse else np.arange(n)
    t0 = time.perf_counter()
    r = bvt.models.var_sv(Y0[:, order], Y[:, order], p, model=model, nsim=nsim, burnin=burnin,
                          rng=seed)
    if reverse:                                # back to the order given
        r.Sig_mean = r.Sig_mean[:, rev][:, :, rev]
    res.append(r)
    print(f'{model}, {"reverse" if reverse else "given"} order: '
          f'{time.perf_counter() - t0:.0f} s')


# ---- how much reversing the order changes each correlation ----
def corrpath(S, i, j):
    return S[:, i, j] / np.sqrt(S[:, i, i] * S[:, j, j])


prs = list(itertools.combinations(range(n), 2))
gap = np.array([[np.mean(np.abs(corrpath(res[0].Sig_mean, i, j) - corrpath(res[1].Sig_mean, i, j))),
                 np.mean(np.abs(corrpath(res[2].Sig_mean, i, j) - corrpath(res[3].Sig_mean, i, j)))]
                for i, j in prs])
print('\naverage absolute change in each correlation when the order is reversed')
print(f"{'':<40s} {'Cholesky':>10s} {'order-invariant':>16s}")
for (i, j), g in zip(prs, gap):
    print(f'{names[i] + ", " + names[j]:<40s} {g[0]:10.3f} {g[1]:16.3f}')
print('\nThe order-invariant model is the same model in both orders, so its column is\n'
      'Monte Carlo error. Correlations are computed from the posterior mean of Sigma_t.')

# ---- the correlation that changes most under the Cholesky model ----
i, j = prs[int(np.argmax(gap[:, 0]))]
lbl = ['Cholesky, order given', 'Cholesky, reverse order', 'order-invariant, order given',
       'order-invariant, reverse order']
sty = ['-', '--', '-', ':']
col = [(0, .447, .741), (.85, .325, .098), (.62, .62, .62), (0, 0, 0)]
wid = [1, 1, 2.6, 1.4]
fig, ax = plt.subplots(figsize=(7.8, 3.8))
h = [None] * 4
for q in [2, 3, 1, 0]:
    h[q], = ax.plot(corrpath(res[q].Sig_mean, i, j), sty[q], color=col[q], linewidth=wid[q])
ax.legend(h, lbl, loc='best', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title(f'Correlation between the {names[i]} and {names[j]} equations')
ax.set_xlabel('observation')
fig.tight_layout()
plt.show()
