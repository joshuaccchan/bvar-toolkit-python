"""your_data - impose your own sign and ranking restrictions on a VAR

Set the file, the columns, the lag length and the restrictions below, then run the script.
It estimates a VAR under the asymmetric conjugate prior of Chan (2022), with the two
shrinkage hyperparameters chosen by maximizing the marginal likelihood, draws candidate
rotations until the requested number are admissible under bvt.structural.sign_assign, and
reports the impulse responses with 68 percent credible bands. The same candidates are also
tested by the accept-reject rule of Rubio-Ramirez, Waggoner and Zha (2010), so the run
reports what each rule accepts.

Restrictions are given per shock as two lists of positions in cols, counted from 0: the
variables whose impact response must be positive, and those whose must be negative. Every
other response is free. A ranking restriction is a row [shock, a, b] meaning that for that
shock the impact response of variable a is at most the impact response of variable b.

The restrictions must satisfy Assumption 2 of Chan, Matthes and Yu (2026), which separates
every pair of shocks either by their signs or by a ranking restriction. The script checks it
before it draws.

The defaults restrict three shocks in a five-variable VAR and take a few seconds. The
tutorial itself restricts eight shocks in 35 variables.

Give cols as column names, which the first row of the file must hold. The selected columns
must have no missing values inside the sample; rows missing at either end are dropped.

Usage, from the root of the repository:  python tutorials/sign_restrictions/your_data.py
"""

import math
import os
import tempfile
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bvar_toolkit as bvt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- settings ----
file = os.path.join(REPO, 'tutorials', 'sv_specification', 'macro5_Q.csv')
cols = ['GDPC1', 'PCECTPI', 'FEDFUNDS', 'UNRATE', 'NFCI']
label = ['GDP growth', 'PCE inflation', 'Fed funds', 'Unemployment', 'NFCI']
datecol = 'Date'         # the column holding the dates; '' to skip
p = 4                    # lags
n0 = 8                   # rows used as initial conditions
horizon = 20             # horizons reported, counting the impact period
nkeep = 100              # admissible draws to collect
nbatch = 500             # posterior draws per batch, one candidate rotation each
levels = []              # columns, from 0, whose first own lag gets prior mean one
ridge = 0                # added to the posterior precision in the marginal likelihood
seed = 1
outdir = tempfile.gettempdir()   # where the report goes; '' for none

shock = ['demand', 'supply', 'monetary']
pos = [[0, 1, 2],        # demand:   output, prices and the policy rate rise
       [0],              # supply:   output rises
       [2, 3]]           # monetary: the policy rate and unemployment rise
neg = [[3],              # demand:   unemployment falls
       [1, 2, 3],        # supply:   prices, the policy rate and unemployment fall
       [0, 1]]           # monetary: output and prices fall
ranking = [[2, 0, 1]]    # under shock 2, variable 0 rises by no more than variable 1:
                         # output falls by at least as much as prices under a monetary
                         # contraction

# ---- data ----
raw = pd.read_csv(file, float_precision='round_trip')
sel = raw[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
keep = np.all(np.isfinite(sel), axis=1)
assert keep.any(), 'every row of the selected columns has a missing value'
lo, hi = np.flatnonzero(keep)[[0, -1]]
assert keep[lo:hi + 1].all(), 'the selected columns have missing values inside the sample'
data = sel[lo:hi + 1]
nobs, n = data.shape
m = len(shock)
assert len(pos) == m and len(neg) == m, 'pos and neg need one entry per shock'
assert len(label) == n, 'label needs one entry per column'
Y0, Y = data[:n0], data[n0:]
_, Z = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)

dates = pd.to_datetime(raw[datecol].iloc[lo:hi + 1], errors='coerce') if datecol else None
if dates is not None and dates.notna().all():
    first, last = dates.iloc[0], dates.iloc[-1]
    print(f'\n{n} variables, {nobs} observations ({first.year}Q{first.quarter} to '
          f'{last.year}Q{last.quarter}), {p} lags')
else:
    print(f'\n{n} variables, {nobs} observations, {p} lags')

# ---- the restrictions ----
S = np.full((n, m), np.nan)
for i in range(m):
    S[pos[i], i] = 1
    S[neg[i], i] = -1
ranking = np.array(ranking, dtype=int).reshape(-1, 3)
kr = int(np.bincount(ranking[:, 0], minlength=m).max()) if ranking.size else 1
Rineq = np.zeros((m, n, kr))
cnt = np.zeros(m, dtype=int)
Rrows, Ridx = [], []
for s, a, b in ranking:
    Rineq[s, a, cnt[s]] = 1
    Rineq[s, b, cnt[s]] = -1
    Rrows.append(Rineq[s, :, cnt[s]].copy())
    Ridx.append(s)
    cnt[s] += 1
Rrows, Ridx = np.array(Rrows).reshape(-1, n), np.array(Ridx, dtype=int)
print(f'{int(np.sum(~np.isnan(S)))} sign restrictions and {len(Ridx)} ranking restrictions '
      f'over {m} shocks')

# Every pair of shocks must be separated, or one column can serve two shocks and the
# assignment is not defined. sign_assign runs once per candidate and does not check it.
sep, unsep = bvt.structural.check_separable(S, Rineq)
if not sep:
    raise ValueError(f'shocks {unsep[0, 0]} and {unsep[0, 1]} are not separable: Assumption 2 '
                     'of Chan, Matthes and Yu (2026) needs either one variable restricted with '
                     'the same sign under both and one with opposite signs, or a ranking '
                     'restriction on the same two variables pointing opposite ways')

# ---- the prior, with its shrinkage chosen by marginal likelihood ----
ml_opt, kappa = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, [.04, .0016], 'redu', levels, ridge=ridge)
sig2 = bvt.priors.resid_var_ar4(Y0, Y)
prior = bvt.priors.acp_redu(n, p, kappa, sig2, levels)
print(f'shrinkage by marginal likelihood: kappa1 {kappa[0]:.4g}, kappa2 {kappa[1]:.4g}, '
      f'log ML {ml_opt:.1f}')

# ---- draw until nkeep rotations are admissible ----
rng = np.random.default_rng(seed)
resp = np.zeros((nkeep, n, m, horizon))
nsat = ntotal = nstrict = 0
t0 = time.perf_counter()
while nsat < nkeep:
    alp, beta, sig = bvt.samplers.acp_theta_sig(Y0, Y, p, prior, nbatch, rng=rng)
    Btilde, Sigtilde = bvt.structural.reduced_form(alp, beta, sig)
    for isim in range(nbatch):
        ntotal += 1
        Q = bvt.structural.qr_sign(rng.standard_normal((n, n)))[0]
        L = np.linalg.cholesky(Sigtilde[isim]) @ Q
        nstrict += bvt.structural.sign_restrict(L, S, Rrows, Ridx)[0]
        ok, Lo = bvt.structural.sign_assign(L, S, Rineq, kr, rng=rng)
        if ok and nsat < nkeep:
            B = Btilde[isim].reshape(n, n * p + 1).T
            resp[nsat] = bvt.structural.irf_redu(B[1:], Lo, horizon, m)
            nsat += 1
print(f'{nsat} admissible draws from {ntotal} candidates, {ntotal / nsat:.0f} each, in '
      f'{time.perf_counter() - t0:.1f} seconds')
print(f'the accept-reject rule accepted {nstrict} of the same candidates')

# ---- the impulse responses ----
lo68 = np.quantile(resp, .16, axis=0, method='hazen')
md = np.median(resp, axis=0)
hi68 = np.quantile(resp, .84, axis=0, method='hazen')

print('\nimpact response to a one-standard-deviation shock, posterior median')
print(f"  {'variable':18s}" + ''.join(f'{s:>16s}' for s in shock))
for i in range(n):
    print(f'  {label[i][:18]:18s}' + ''.join(f'{md[i, s, 0]:16.3f}' for s in range(m)))

nrow = 1 + (n > 3)
x = np.arange(horizon)
for s in range(m):
    fig, axes = plt.subplots(nrow, math.ceil(n / nrow), figsize=(7.8, nrow * 1.8 + 0.9),
                             squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.set_visible(False)
            continue
        bvt.util.shaded_band(x, lo68[i, s], hi68[i, s], .85, ax=ax)
        ax.plot(x, md[i, s], 'k--', linewidth=1.1)
        ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
        ax.set_xlim(0, horizon - 1)
        ax.set_title(label[i], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{shock[s]} shock')
    fig.tight_layout()

# ---- the report ----
if outdir:
    rows = [(shock[s], cols[i], h, lo68[i, s, h], md[i, s, h], hi68[i, s, h])
            for s in range(m) for i in range(n) for h in range(horizon)]
    irf = pd.DataFrame(rows, columns=['shock', 'variable', 'horizon', 'p16', 'median', 'p84'])
    meta = dict(file=file, columns=cols, p=p, n0=n0, n=n, T=Y.shape[0], shocks=shock,
                horizon=horizon, nkeep=nkeep, candidates=ntotal, accepted_strict=int(nstrict),
                kappa=kappa, log_ml=ml_opt, ridge=ridge, seed=seed)
    bvt.util.report('sign_restrictions', {'irf': irf}, meta, outdir)

if matplotlib.get_backend().lower() != 'agg':
    plt.show()
