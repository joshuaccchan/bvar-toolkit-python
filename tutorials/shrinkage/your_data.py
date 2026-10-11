"""your_data - choose the shrinkage hyperparameters of a BVAR on your data

Set the file, the columns, their names, which of them are nonstationary and the lag lengths
below, then run the script. It chooses the own-lag and other-lag shrinkage kappa2 and kappa3
of the asymmetric conjugate prior by maximizing the closed-form marginal likelihood, compares
the result with the best symmetric prior (kappa2 = kappa3) and with the subjective values
kappa2 = 0.04 and kappa3 = 0.0016, repeats the choice for each lag length up to pmax, plots
the marginal likelihood around the optimum, forecasts H periods past the end of the sample
under each prior, and writes everything to csv files and a json file. The prior is set on
the structural-form coefficients, as in Section 12.3 of Bayesian Macroeconometrics. The
defaults use the 21-variable dataset of the tutorial.

Give cols as column positions, counted from 0 over every column of the file, or as column
names, which the first row of the file must hold. The selected columns must have no missing
values inside the sample; rows missing at either end are dropped. Setting datecol checks
that the rows kept are evenly spaced. The series must be stationary, transformed as needed;
that choice is separate from nonstationary below, which only centers the prior.

The first n0 rows serve as initial conditions: at least max(pmax, 4) of them.

Usage, from the root of the repository:  python tutorials/shrinkage/your_data.py
"""

import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bvar_toolkit as bvt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- settings ----
file = os.path.join(REPO, 'data', 'chan2019wp_acp', 'macrodata_Q_2018Q4.csv')
cols = [0, 1, 17, 21, 33, 34, 56, 58, 75, 80, 94, 96, 119, 122, 132, 137, 144, 147, 151, 159, 244]
names = ['GDP', 'Consumption', 'Disposable income', 'Industrial production',
         'Capacity utilization', 'Payroll employment', 'Civilian employment',
         'Unemployment rate', 'Hours', 'Housing starts', 'PCE prices', 'GDP deflator', 'CPI',
         'PPI', 'Real earnings', 'Productivity', '3-month T-bill', '10-year yield',
         'Baa spread', 'Real M1', 'S&P 500']
datecol = None         # the column holding the dates, by position or name; None to skip the check
nonstationary = []     # positions in cols, from 0, whose first own lag gets prior mean one
p = 4                  # lags
pmax = 8               # longest lag length in the scan
n0 = 8                 # rows used as initial conditions
H = 4                  # forecast horizons past the end of the sample; 0 for none
nsim_f = 5000          # posterior draws behind the forecasts
seed = 1
outdir = tempfile.gettempdir()   # where the report goes; '' for none

# ---- data ----
if all(isinstance(c, (int, np.integer)) for c in cols):
    raw = pd.read_csv(file, header=None, float_precision='round_trip')   # positions count every column
    sel = raw.iloc[:, cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
else:
    raw = pd.read_csv(file, float_precision='round_trip')
    sel = raw[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
dates = None if datecol is None else (raw.iloc[:, datecol] if isinstance(datecol, int) else raw[datecol])
assert len(names) == len(cols), 'names and cols must have the same length'
assert n0 >= max(pmax, 4), 'n0 must be at least max(pmax, 4)'
keep = np.all(np.isfinite(sel), axis=1)
assert keep.any(), 'every row of the selected columns has a missing value'
lo, hi = np.flatnonzero(keep)[[0, -1]]
assert keep[lo:hi + 1].all(), 'the selected columns have missing values inside the sample'
data = sel[lo:hi + 1]
rows_kept = [int(lo), int(hi)]
flat = np.std(data, axis=0) == 0
assert not flat.any(), 'these series are constant: ' + ', '.join(np.array(names)[flat])
if dates is not None:
    d = dates.iloc[lo:hi + 1]
    num = pd.to_numeric(d, errors='coerce')
    if num.notna().all():
        step = np.diff(num.to_numpy(dtype=float))
        even = np.all(step > 0) and np.max(np.abs(step - np.median(step))) <= 1e-6 * max(1, abs(np.median(step)))
    else:
        step = np.diff(pd.to_datetime(d).to_numpy()) / np.timedelta64(1, 'D')
        even = np.all(step > 0) and np.max(np.abs(step - np.median(step))) <= 3   # months vary by up to 3 days
    assert even, 'the dates of the rows kept are not evenly spaced'
Y0, Y = data[:n0], data[n0:]
T, n = Y.shape
print(f'\n{n} variables, {T} observations after {n0} initial conditions')

# ---- the three priors at p lags ----
_, Z = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
sig2 = bvt.priors.resid_var_ar4(Y0, Y)
k_subj = np.array([.04, .0016, 1, 100])
ml_a, ka = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, [.04, .04], 'stru', nonstationary)
ml_s, ks = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, None, 'stru', nonstationary, symmetric=True)
ml_0 = bvt.ml.acp(p, Y, Z, bvt.priors.acp_stru(n, p, k_subj, sig2, nonstationary))
    # the symmetric optimum is feasible under the asymmetric prior, so a search started
    # there should reach the same maximum
ml_b, kb = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, [ks[0], ks[0]], 'stru', nonstationary)
gap = ml_b - ml_a
if gap > 0:
    ml_a, ka = ml_b, kb
print(f'\nlog marginal likelihood at p = {p}')
print(f"{'':42s} {'kappa2':>10s} {'kappa3':>10s} {'log ML':>10s} {'difference':>10s}")
print(f"{'symmetric, chosen by marginal likelihood':42s} {ks[0]:10.4f} {ks[1]:10.4f} {ml_s:10.1f} {ml_s - ml_a:10.1f}")
print(f"{'subjective':42s} {.04:10.4f} {.0016:10.4f} {ml_0:10.1f} {ml_0 - ml_a:10.1f}")
print(f"{'asymmetric, chosen by marginal likelihood':42s} {ka[0]:10.4f} {ka[1]:10.4f} {ml_a:10.1f} {0:10.1f}")
print('kappa2 and kappa3 are the own-lag and other-lag shrinkage of Section 12.3 of\n'
      'Bayesian Macroeconometrics; bvt.priors.acp_opt_kappa returns them as kappa[0] and kappa[1].')
print('two starting values for the asymmetric search, (0.04, 0.04) and the symmetric')
print(f'optimum ({ks[0]:.4f}, {ks[1]:.4f}), reach log marginal likelihoods {abs(gap):.4f} apart')
print('the estimates condition on these hyperparameters: their uncertainty is left out')

# ---- the lag length ----
print('\nlag length, with kappa2 and kappa3 chosen again for each')
print(f"{'p':>4s} {'kappa2':>10s} {'kappa3':>10s} {'log ML':>12s} {'symmetric ML':>14s}")
lml, lsym = np.zeros(pmax), np.zeros(pmax)
kap, kaps = np.zeros((pmax, 2)), np.zeros((pmax, 2))
for pp in range(1, pmax + 1):
    _, Zp = bvt.util.build_lags(np.vstack([Y0[-pp:], Y]), pp)
    lml[pp - 1], kp = bvt.priors.acp_opt_kappa(Y0, Y, Zp, pp, [.04, .04], 'stru', nonstationary)
    lsym[pp - 1], kq = bvt.priors.acp_opt_kappa(Y0, Y, Zp, pp, None, 'stru', nonstationary, symmetric=True)
    kap[pp - 1], kaps[pp - 1] = kp[:2], kq[:2]
    print(f'{pp:4d} {kp[0]:10.4f} {kp[1]:10.5f} {lml[pp - 1]:12.1f} {lsym[pp - 1]:14.1f}')
pbest = int(np.argmax(lml)) + 1
print(f'highest marginal likelihood at p = {pbest}')
if pbest == pmax:
    print('this is the longest lag length in the scan: raise pmax, and n0 with it,\n'
          'to see whether the marginal likelihood keeps rising')

# ---- forecasts from the end of the sample ----
# Given the coefficients and the error covariance matrix, the h-step forecast of a VAR is
# Gaussian, and bvt.forecast.predictive returns its mean and standard deviation for each
# posterior draw. Drawing one value from each of those normals gives the predictive
# distribution itself; its mean and standard deviation are printed, and the report also
# carries its 68% band under each of the three priors.
pname = ['symmetric', 'subjective', 'asymmetric']
fcast = None
if H > 0:
    Kf = [ks, k_subj, ka]
    ylag = data[-p:][::-1].T                    # most recent lag first
    fm, fs, flo, fhi = (np.zeros((n, H, 3)) for _ in range(4))
    for j in range(3):
        prior = bvt.priors.acp_stru(n, p, Kf[j], sig2, nonstationary)
        rng = np.random.default_rng(seed)       # common random numbers across the priors
        Alp, Beta, Sg = bvt.samplers.acp_theta_sig(Y0, Y, p, prior, nsim_f, rng=rng)
        Bt, St = bvt.structural.reduced_form(Alp, Beta, Sg)
        mu, sd = bvt.forecast.predictive(Bt, St, ylag, H)
        yd = mu + sd * rng.standard_normal(mu.shape)
        q = np.quantile(yd, [.16, .84], axis=0, method='hazen')
        fm[:, :, j] = mu.mean(axis=0)
        fs[:, :, j] = np.sqrt(np.mean(sd ** 2, axis=0) + np.var(mu, axis=0, ddof=1))
        flo[:, :, j], fhi[:, :, j] = q[0], q[1]
    print(f'\nforecasts under the asymmetric prior, from {nsim_f} posterior draws')
    print('posterior mean, with the standard deviation of the predictive distribution')
    print(f"{'variable':24s}" + ''.join(f'{f"h = {h}":>18s}' for h in range(1, H + 1)))
    for i in range(n):
        print(f'{names[i][:24]:24s}' + ''.join(f'{fm[i, h, 2]:10.2f} ({fs[i, h, 2]:5.2f})' for h in range(H)))
    print('the 68% bands and the other two priors are in the report')
    print('the draws condition on the chosen hyperparameters, so the spread leaves out')
    print('the uncertainty about kappa2 and kappa3')
    iv, ih, ip = np.meshgrid(np.arange(n), np.arange(H), np.arange(3), indexing='ij')
    iv, ih, ip = iv.ravel(order='F'), ih.ravel(order='F'), ip.ravel(order='F')
    fcast = pd.DataFrame({'prior': np.array(pname)[ip], 'variable': np.array(names)[iv], 'h': ih + 1,
                          'mean': fm.ravel(order='F'), 'sd': fs.ravel(order='F'),
                          'lo68': flo.ravel(order='F'), 'hi68': fhi.ravel(order='F')})

# ---- the marginal likelihood around the optimum ----
    # a logarithmic grid from a tenth to ten times the optimum, widened to show the
    # symmetric and subjective priors
r2 = [min(ka[0] / 10, ks[0], .04) / 1.5, ka[0] * 10]
r3 = [min(ka[1] / 10, .0016) / 1.5, max(ka[1] * 10, ks[1]) * 1.5]
g2 = np.logspace(np.log10(r2[0]), np.log10(r2[1]), 41)
g3 = np.logspace(np.log10(r3[0]), np.log10(r3[1]), 41)
K2, K3 = np.meshgrid(g2, g3)
L = np.array([bvt.ml.acp(p, Y, Z, bvt.priors.acp_stru(n, p, [a, b, 1, 100], sig2, nonstationary))
              for a, b in zip(K2.ravel(), K3.ravel())]).reshape(K2.shape)
fig, ax = plt.subplots()
ax.contour(K2, K3, np.exp(L - L.max()), levels=np.arange(1, 10) / 10, linewidths=1)
ax.set_xscale('log')
ax.set_yscale('log')
dlo, dhi = max(g2[0], g3[0]), min(g2[-1], g3[-1])
if dlo < dhi:
    ax.plot([dlo, dhi], [dlo, dhi], 'k--', linewidth=0.8)
ax.plot(ka[0], ka[1], 'k*', mfc='w', ms=13, label='asymmetric prior')
ax.plot(ks[0], ks[1], 'ko', mfc='k', label='symmetric prior')
ax.plot(.04, .0016, 'ks', mfc='k', label='subjective prior')
ax.set_xlim(r2)
ax.set_ylim(r3)
ax.legend(loc='best', frameon=False)
ax.set_xlabel(r'$\kappa_2$ (own lags), smaller is tighter')
ax.set_ylabel(r'$\kappa_3$ (other lags), smaller is tighter')
ax.set_title(f'Marginal likelihood relative to its maximum, p = {p}')
ax.spines[['top', 'right']].set_visible(False)

# ---- the report ----
# One row per prior at p lags and per lag length in the scan, the forecasts in their own
# file, and the settings in the json file.
if outdir:
    models = pd.DataFrame({
        'block': ['prior comparison'] * 3 + ['lag length'] * (2 * pmax),
        'prior': pname + ['asymmetric'] * pmax + ['symmetric'] * pmax,
        'p': [p] * 3 + list(range(1, pmax + 1)) * 2,
        'kappa2': [ks[0], k_subj[0], ka[0], *kap[:, 0], *kaps[:, 0]],
        'kappa3': [ks[1], k_subj[1], ka[1], *kap[:, 1], *kaps[:, 1]],
        'log_ML': [ml_s, ml_0, ml_a, *lml, *lsym]})
    meta = dict(file=file, columns=cols, names=names, nonstationary=nonstationary, p=p,
                pmax=pmax, n0=n0, n=n, T=T, rows_kept=rows_kept, H=H, nsim_f=nsim_f, seed=seed,
                kappa_asymmetric=ka, kappa_symmetric=ks, kappa_subjective=k_subj)
    out = {'models': models}
    if fcast is not None:
        out['forecasts'] = fcast
    bvt.util.report('shrinkage_report', out, meta, outdir)

if matplotlib.get_backend().lower() != 'agg':
    plt.show()
