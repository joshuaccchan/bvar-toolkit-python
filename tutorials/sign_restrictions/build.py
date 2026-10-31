"""build - regenerate the figures and numbers of tutorials/sign_restrictions/README.md

The 35-variable application of Chan, Matthes and Yu (2026): eight structural shocks
set-identified by sign restrictions on the impact responses and by two ranking restrictions,
on US quarterly data from 1983Q1 to 2019Q4. The model is a homoskedastic VAR with five lags
under the asymmetric conjugate prior of Chan (2022), whose two shrinkage hyperparameters are
chosen by maximizing the marginal likelihood, evaluated with the ridge of 1e-6 that this
package adds to the posterior precision.

A candidate rotation is admissible when every shock has at least one column of the rotated
Cholesky factor that satisfies its restrictions, and bvt.structural.sign_assign then draws
one assignment from those available. The accept-reject rule of Rubio-Ramirez, Waggoner and
Zha (2010), bvt.structural.sign_restrict, instead requires column i to satisfy shock i.
Both are applied to every candidate here, so the run reports what each rule accepts out of
the same draws.

The restrictions are written out below in a readable form and then checked against the
matrices in the package's own driver, which this script reads.

This build keeps 100 admissible draws where the paper keeps 1,000.

The posterior summaries of the impulse responses go to irf_bands.npz and the figures are
written next to this file. Everything printed also goes to a log in the temporary directory.

Usage, from the root of the repository:  python tutorials/sign_restrictions/build.py
"""

import datetime
import math
import os
import platform
import re
import sys
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

TDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TDIR))
PKG = os.path.join(REPO, 'data', 'chan_matthes_yu2026_qe_svarsign')


class Tee:
    """Print to the screen and to a log file."""

    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.screen = sys.stdout

    def write(self, s):
        self.screen.write(s)
        self.file.write(s)
        self.flush()

    def flush(self):
        self.screen.flush()
        self.file.flush()


def package_restrictions(path):
    """The restrictions stated in the package's driver: the block from 'demand = [' to
    'start_time = clock', read as the MATLAB assignments it holds."""
    src = open(path, encoding='utf-8').read()
    i0, i1 = src.find('demand = ['), src.find('start_time = clock')
    assert i0 > 0 and i1 > i0, 'the restriction block was not found in the driver'
    block = re.sub(r'\.\.\.\r?\n', ' ', src[i0:i1])            # join continued lines
    block = '\n'.join(line.split('%', 1)[0] for line in block.splitlines())
    vectors = {name: [float(v) for v in re.split(r'[,\s]+', body.strip())]
               for name, body in re.findall(r"(\w+)\s*=\s*\[([^\]]*)\]'", block)}
    assert all(len(v) == 35 for v in vectors.values()), 'a restriction vector is not of length 35'
    names = [s.strip() for s in re.search(r'\bS\s*=\s*\[([^\]]*)\]', block).group(1).split(',')]
    S = np.column_stack([vectors[nm] for nm in names])
    k = int(re.search(r'\bk\s*=\s*(\d+)', block).group(1))
    Rineq = np.zeros((S.shape[1], S.shape[0], k))
    for i, j, layer, v in re.findall(r'Rineq\((\d+),(\d+),(\d+)\)\s*=\s*(-?\d+)', block):
        Rineq[int(i) - 1, int(j) - 1, int(layer) - 1] = float(v)
    return S, Rineq


def main():
    log = os.path.join(tempfile.gettempdir(), 'bvar_toolkit_python_sign_restrictions_build_log.txt')
    sys.stdout = Tee(log)
    print(f'tutorials/sign_restrictions/build.py, {datetime.datetime.now():%Y-%m-%d %H:%M}, '
          f'Python {platform.python_version()}, NumPy {np.__version__}, '
          f'SciPy {scipy.__version__}, bvar_toolkit {bvt.__version__}')
    t_all = time.perf_counter()

    # ---- settings ----
    p, n0, nkeep, nbatch, horizon = 5, 8, 100, 1000, 36
    seed, ridge, k0 = 20260920, 1e-6, [.04, .0016]

    # ---- data ----
    # The driver reads the block B35:AJ182 of the csv: 148 quarters of 35 series
    raw = pd.read_csv(os.path.join(PKG, 'data35_1975.csv'), header=None, dtype=str)
    data = raw.iloc[34:182, 1:36].astype(float).to_numpy()
    names = [s.strip() for s in raw.iloc[0, 1:36]]
    nobs, n = data.shape
    assert (nobs, n) == (148, 35), 'the data block is not 148 x 35'

    def qlab(i):
        return f'{1983 + i // 4}Q{i % 4 + 1}'
    Y0, Y = data[:n0], data[n0:]
    T = Y.shape[0]
    _, Z = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
    idx_ns = np.arange(n)                           # every series enters in levels
    print(f'\n{n} variables, {nobs} quarters ({qlab(0)} to {qlab(nobs - 1)}), {p} lags')
    print(f'{T} observations after {n0} initial conditions')

    # ---- the prior, with its shrinkage chosen by marginal likelihood ----
    t0 = time.perf_counter()
    ml_opt, kappa = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, k0, 'redu', idx_ns, ridge=ridge)
    sig2 = bvt.priors.resid_var_ar4(Y0, Y)
    prior = bvt.priors.acp_redu(n, p, kappa, sig2, idx_ns)
    print(f'\nshrinkage by marginal likelihood: kappa1 {kappa[0]:.4g} (own lags), '
          f'kappa2 {kappa[1]:.4g} (cross lags)')
    print(f'log marginal likelihood {ml_opt:.1f}, chosen in '
          f'{(time.perf_counter() - t0) / 60:.1f} minutes')

    # ---- the restrictions ----
    shock = ['demand', 'investment', 'financial', 'monetary', 'government spending',
             'technology', 'labor supply', 'wage bargaining']
    m = len(shock)
    # positions of the variables, from 0, whose impact response rises and falls
    pos = [[0, 9, 10, 11, 12, 13, 19, 20, 24, 25, 29],
           [0, 9, 10, 11, 12, 13, 19, 20, 24, 25, 29],
           [0, 9, 10, 11, 12, 13, 19, 20, 24, 25, 29, 33],
           [18, 24, 25, 26, 27, 28, 29, 30, 31],
           [0, 6, 8, 9, 10, 11, 12, 13, 24, 25, 29],
           [0, 1, 3, 14, 15, 16],
           [0, 18],
           [0]]
    neg = [[18], [18, 33], [18], [0, 9, 10, 11, 12, 13, 17, 19, 20, 33], [7, 18],
           [9, 10, 11, 12, 13, 18], [9, 10, 11, 12, 13, 14], [9, 10, 11, 12, 13, 14, 18]]
    S = np.full((n, m), np.nan)
    for i in range(m):
        S[pos[i], i] = 1
        S[neg[i], i] = -1

    # Each row is a linear combination of impact responses required to be <= 0 for that
    # shock's column. Variables 0, 3 and 6 are output, nonresidential investment and
    # government spending.
    kr = 2
    Rineq = np.zeros((m, n, kr))
    Rineq[0, [0, 3], 0] = [-1, 1]    # demand:     investment rises by less than output
    Rineq[1, [0, 3], 0] = [1, -1]    # investment: investment rises by more
    Rineq[2, [0, 3], 0] = [1, -1]    # financial:  investment rises by more
    Rineq[0, [0, 6], 1] = [-1, 1]    # demand:     government spending rises by less
    Rineq[1, [0, 6], 1] = [-1, 1]    # investment: government spending rises by less
    Rineq[2, [0, 6], 1] = [-1, 1]    # financial:  government spending rises by less
    Rineq[4, [0, 6], 1] = [1, -1]    # government: government spending rises by more

    S_pkg, R_pkg = package_restrictions(os.path.join(PKG, 'main_35VAR_Figs2to10_except5.m'))
    assert np.array_equal(S, S_pkg, equal_nan=True), \
        "the sign restrictions differ from the package's driver"
    assert np.array_equal(Rineq, R_pkg), "the ranking restrictions differ from the package's driver"
    print(f'\n{int(np.sum(~np.isnan(S)))} sign restrictions and '
          f'{int(np.sum(np.any(Rineq != 0, axis=1)))} ranking restrictions over {m} shocks,')
    print('checked against main_35VAR_Figs2to10_except5.m')

    sep, unsep = bvt.structural.check_separable(S, Rineq)
    assert sep, f'the restrictions leave {len(unsep)} pairs of shocks unseparated'
    _, by_rank = bvt.structural.check_separable(S)
    print(f'all {m * (m - 1) // 2} pairs of shocks are separated, {len(by_rank)} of them only by '
          f'the ranking restrictions')

    # the same ranking restrictions in the form sign_restrict takes: one row per
    # restriction, with the column of L it applies to
    Rrows, Ridx = [], []
    for i in range(m):
        for j in range(kr):
            if np.any(Rineq[i, :, j] != 0):
                Rrows.append(Rineq[i, :, j])
                Ridx.append(i)
    Rrows, Ridx = np.array(Rrows), np.array(Ridx)

    # ---- draw until 100 rotations are admissible ----
    print(f'\ndrawing until {nkeep} rotations are admissible, {nbatch} posterior draws per batch')
    rng = np.random.default_rng(seed)
    resp = np.zeros((nkeep, n, m, horizon))
    nsat = ntotal = nstrict = ibatch = 0
    t0 = time.perf_counter()
    while nsat < nkeep:
        ibatch += 1
        alp, beta, sig = bvt.samplers.acp_theta_sig(Y0, Y, p, prior, nbatch, rng=rng)
        Btilde, Sigtilde = bvt.structural.reduced_form(alp, beta, sig)
        for isim in range(nbatch):
            ntotal += 1
            L0 = np.linalg.cholesky(Sigtilde[isim])
            Q, _ = bvt.structural.qr_sign(rng.standard_normal((n, n)))
            L = L0 @ Q
            nstrict += bvt.structural.sign_restrict(L, S, Rrows, Ridx)[0]
            ok, Lo = bvt.structural.sign_assign(L, S, Rineq, kr, rng=rng)
            if ok and nsat < nkeep:
                B = Btilde[isim].reshape(n, n * p + 1).T
                resp[nsat] = bvt.structural.irf_redu(B[1:], Lo, horizon, m)
                nsat += 1
        if ibatch % 25 == 0:
            print(f'  {nsat} admissible out of {ntotal} candidates, '
                  f'{(time.perf_counter() - t0) / 60:.1f} minutes')
    t_draw = time.perf_counter() - t0

    print(f'\n{nsat} admissible draws from {ntotal} candidate rotations, '
          f'{ntotal / nsat:.0f} candidates each,')
    print(f'in {t_draw / 60:.1f} minutes. The paper reports about 5,500 candidates per draw.')
    print(f'the accept-reject rule of Rubio-Ramirez, Waggoner and Zha accepted {nstrict} of the '
          f'same {ntotal}.')
    print(f'it tests one assignment of shocks to columns out of the '
          f'{math.perm(n, m):.1e} available;')
    print('both rules are free to flip the sign of a column, so the sign is not the difference.')

    # ---- the impulse responses ----
    lo = np.quantile(resp, .16, axis=0, method='hazen')
    md = np.median(resp, axis=0)
    hi = np.quantile(resp, .84, axis=0, method='hazen')
    np.savez(os.path.join(TDIR, 'irf_bands.npz'), lo=lo, md=md, hi=hi, vars=np.array(names),
             shock=np.array(shock), kappa=kappa, ml_opt=ml_opt, nsat=nsat, ntotal=ntotal,
             nstrict=nstrict, horizon=horizon, p=p, seed=seed)

    show = [0, 10, 24, 3, 18, 14]
    label = ['GDP', 'PCE price index', 'Federal funds rate', 'Nonresidential investment',
             'Unemployment rate', 'Real compensation']
    short = ['GDP', 'prices', 'fed funds', 'investment', 'unemployment', 'compensation']
    print('\nimpact response to a one-standard-deviation shock, posterior median')
    print(f"  {'shock':21s}" + ''.join(f'{s:>13s}' for s in short))
    for i_s in range(m):
        print(f'  {shock[i_s][:21]:21s}' + ''.join(f'{md[v, i_s, 0]:13.3f}' for v in show))
    print('\nimpact gaps between posterior medians: investment minus output, financial '
          f'{md[3, 2, 0] - md[0, 2, 0]:.3f}, demand {md[3, 0, 0] - md[0, 0, 0]:.3f}; '
          f'government spending minus output, government spending shock '
          f'{md[6, 4, 0] - md[0, 4, 0]:.3f}')

    for i_s in [3, 2]:                  # the monetary and financial shocks get a figure each
        fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.6))
        x = np.arange(horizon)
        for j, ax in enumerate(axes.ravel()):
            bvt.util.shaded_band(x, lo[show[j], i_s], hi[show[j], i_s], .85, ax=ax)
            ax.plot(x, md[show[j], i_s], 'k--', linewidth=1.1)
            ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
            ax.set_xlim(0, horizon - 1)
            ax.set_title(label[j], fontsize=10)
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'{shock[i_s]} shock')
        fig.tight_layout()
        fig.savefig(os.path.join(TDIR, f"fig_irf_{re.sub(r'[ ]+', '_', shock[i_s])}.png"), dpi=150)
        plt.close(fig)

    print(f'\nbuild finished in {(time.perf_counter() - t_all) / 60:.1f} minutes')
    sys.stdout.file.close()
    sys.stdout = sys.stdout.screen


if __name__ == '__main__':
    main()
