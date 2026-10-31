"""build - regenerate the figures and numbers of tutorials/mixed_frequency/README.md

The model and priors are those of the first application of Chan, Poon and Zhu (2023), a VAR
with common stochastic volatility, estimated here on monthly data: five FRED-MD series and two
quarterly FRED-QD series, real GDP and real private fixed investment, from the August 2026
vintages. Part 1 builds the data and the restrictions, Part 2 estimates the model, Part 3
reports the monthly estimates and compares monthly GDP with the Brave-Butters-Kelley series.
The figures are written next to this file and everything printed goes to a log in the
temporary directory.

Usage, from the root of the repository:  python tutorials/mixed_frequency/build.py
"""

import datetime
import os
import platform
import sys
import tempfile
import time

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

TDIR = os.path.dirname(os.path.abspath(__file__))


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


def months(s):
    """Dates written yyyy-MM, as numpy months."""
    return pd.to_datetime(s, format='%Y-%m').to_numpy().astype('datetime64[M]')


def month(s):
    return np.datetime64(s, 'M')


def years(t):
    return t.astype('datetime64[Y]').astype(int) + 1970


def mlab(t):
    i = int(np.datetime64(t, 'M').astype(int))
    return f'{1970 + i // 12}:{i % 12 + 1:02d}'


def qlab(t):
    i = int(np.datetime64(t, 'M').astype(int))
    return f'{1970 + i // 12}Q{i % 12 // 3 + 1}'


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def rms(a, b):
    return np.sqrt(np.mean((a - b)**2))


def main():
    log = os.path.join(tempfile.gettempdir(), 'bvar_toolkit_python_mixed_frequency_build_log.txt')
    sys.stdout = Tee(log)
    print(f'tutorials/mixed_frequency/build.py, {datetime.datetime.now():%Y-%m-%d %H:%M}, '
          f'Python {platform.python_version()}, NumPy {np.__version__}, '
          f'SciPy {scipy.__version__}, bvar_toolkit {bvt.__version__}')
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    #  Part 1. The data and the restrictions
    # ------------------------------------------------------------------
    D = pd.read_csv(os.path.join(TDIR, 'mf_data.csv'), float_precision='round_trip')
    dlev = months(D['date'])
    d = dlev[1:]                                  # the months of the growth rates
    vnames = ['IP', 'CPI', 'Unemployment', 'Payrolls', 'Hours', 'GDP', 'Investment']
    n, T = len(vnames), d.size
    iq = [5, 6]                                   # the quarterly series

    # IP, CPI and payrolls in 100 x log changes, with a restriction across each gap
    G, Mg, zg = bvt.util.dlog_gaps(D[['INDPRO', 'CPIAUCSL', 'PAYEMS']].to_numpy())
    X = np.column_stack([G[:, 0], G[:, 1], D['UNRATE'].to_numpy()[1:], G[:, 2],
                         D['AWHMAN'].to_numpy()[1:] / 10, np.full((T, 2), np.nan)])
    # GDP and investment: 100 x log change between quarters, in the last month of the quarter
    Q = D[['GDPC1', 'FPIx']].to_numpy()
    for j in range(2):
        iobs = np.flatnonzero(~np.isnan(Q[:, j]))
        X[iobs[1:] - 1, iq[j]] = 100 * np.diff(np.log(Q[iobs, j]))
    Mq, zq, Y, dropped = bvt.util.mm_constraint(X, np.isin(np.arange(n), iq))
    assert not dropped.any(), 'every quarterly value should have its five months in the sample'
    # the rows of dlog_gaps act on the three series it was given; move them to columns 0, 1, 3
    Mg = Mg.tocoo()
    cols = (Mg.col // 3) * n + np.array([0, 1, 3])[Mg.col % 3]
    Mg = sp.csc_array((np.ones(Mg.nnz), (Mg.row, cols)), shape=(Mg.shape[0], T * n))
    M = sp.vstack([Mq, Mg], format='csc')
    z = np.concatenate([zq, zg])

    # the Minnesota scales of the quarterly series: 9/19 of the AR(4) residual variance of the
    # quarterly growth rates, the monthly variance that gives it under the aggregation
    sig2 = np.full(n, np.nan)
    for j in iq:
        g = X[~np.isnan(X[:, j]), j]
        Zq = np.column_stack([np.ones(g.size - 4), g[3:-1], g[2:-2], g[1:-3], g[:-4]])
        e = g[4:] - Zq @ np.linalg.lstsq(Zq, g[4:], rcond=None)[0]
        sig2[j] = 9 / 19 * np.mean(e**2)
    p = 12
    gq = np.flatnonzero(~np.isnan(X[:, 5]))
    print(f'\ndata: {mlab(d[0])} to {mlab(d[-1])}, {T} months, the first {p} as initial '
          f'conditions; {zq.size // 2} quarterly values of GDP and of investment, '
          f'{qlab(d[gq[0]])} to {qlab(d[gq[-1]])}')
    print('missing monthly values: '
          + ', '.join(f'{v} {k}' for v, k in zip(vnames, np.isnan(Y).sum(axis=0))))
    icpi = np.flatnonzero(np.isnan(Y[:, 1]))
    print(f'restrictions across gaps: {zg.size}, CPI over {mlab(d[icpi[0]])}-{mlab(d[icpi[-1]])}')

    # ------------------------------------------------------------------
    #  Part 2. Estimation
    # ------------------------------------------------------------------
    nsim, burnin, seed, hblock = 20000, 2000, 1, 36
    t1 = time.perf_counter()
    res = bvt.models.mfvar_csv(Y, p, M=M, z=z, sig2=sig2, nsim=nsim, burnin=burnin, rng=seed,
                               hblock=hblock, draws=True)
    print(f'\n{nsim} draws after a burn-in period of {burnin}, seed {seed}: '
          f'{(time.perf_counter() - t1) / 60:.1f} minutes')
    print(f'share of the {hblock}-month blocks of h accepted: {res.accept_rate:.3f}')
    print(f'|M*y - z| over the kept draws: max {res.max_resid:.1e}')
    print(f'posterior means: phi {res.phi_mean:.3f}, sigh2 {res.sigh2_mean:.3f}')

    # inefficiency factors of the monthly values of GDP and investment, of h, phi and sigh2
    vm = np.flatnonzero(np.isnan(Y.ravel())) % n
    L = 100
    IF_g = bvt.diag.inefficiency_factor(res.draws.ym[:, vm == 5].astype(float), L)
    IF_i = bvt.diag.inefficiency_factor(res.draws.ym[:, vm == 6].astype(float), L)
    IF_h = bvt.diag.inefficiency_factor(res.draws.h, L)
    IF_p = bvt.diag.inefficiency_factor(np.column_stack([res.draws.phi, res.draws.sigh2]), L)
    print(f'inefficiency factors, median and largest: GDP {np.median(IF_g):.1f} {IF_g.max():.1f}, '
          f'investment {np.median(IF_i):.1f} {IF_i.max():.1f}, '
          f'h {np.median(IF_h):.1f} {IF_h.max():.1f}')
    print(f'inefficiency factors of phi and sigh2: {IF_p[0]:.1f} {IF_p[1]:.1f}')

    # the same sampler with the whole volatility path as one block, briefly, for its acceptance
    # rate
    r1 = bvt.models.mfvar_csv(Y, p, M=M, z=z, sig2=sig2, nsim=1000, burnin=200, rng=seed,
                              hblock=T - p)
    print(f'share of proposals accepted when h is one block of {T - p} months: '
          f'{r1.accept_rate:.3f}')

    # ------------------------------------------------------------------
    #  Part 3. The monthly estimates
    # ------------------------------------------------------------------
    gm, qlo, qhi = res.Y_mean, res.Y_q[:, :, 0], res.Y_q[:, :, 4]
    print('\nmonthly growth, percent: posterior mean (90% band)')
    print(f"{'month':<8s} {'GDP':>22s} {'investment':>22s}")
    for t in np.flatnonzero((d >= month('2020-01')) & (d <= month('2020-12'))):
        print(f'{mlab(d[t]):<8s} {gm[t, 5]:7.2f} ({qlo[t, 5]:5.1f}, {qhi[t, 5]:5.1f}) '
              f'{gm[t, 6]:7.2f} ({qlo[t, 6]:5.1f}, {qhi[t, 6]:5.1f})')
    print('the posterior mean path satisfies the aggregation: max |M*y - z| '
          f'{np.max(np.abs(M @ gm.ravel() - z)):.1e}')
    tq = np.flatnonzero(d == month('2020-06'))[0]
    print(f'2020Q2, 100 x log change: GDP {X[tq, 5]:.2f}, investment {X[tq, 6]:.2f}')

    # the ragged edge: July 2026, with no quarterly value yet
    t = T - 1
    print(f'\n{mlab(d[t])}, before the third-quarter release: GDP {gm[t, 5]:.2f} '
          f'({qlo[t, 5]:.1f}, {qhi[t, 5]:.1f}), investment {gm[t, 6]:.2f} '
          f'({qlo[t, 6]:.1f}, {qhi[t, 6]:.1f})')
    # the values that were never published: CPI inflation in the two months around the missing
    # October 2025 index, whose sum is known, and the October 2025 unemployment rate
    for t in np.flatnonzero(np.isnan(Y[:, 1])):
        print(f'CPI inflation {mlab(d[t])}: {gm[t, 1]:.3f} ({qlo[t, 1]:.3f}, {qhi[t, 1]:.3f})')
    print(f'  their sum, from the published index: {zg[0]:.3f}')
    t = np.flatnonzero(np.isnan(Y[:, 2]))[0]
    print(f'unemployment rate {mlab(d[t])}: {gm[t, 2]:.2f} ({qlo[t, 2]:.1f}, {qhi[t, 2]:.1f}); '
          f'published {mlab(d[t - 1])} {Y[t - 1, 2]:.1f} and {mlab(d[t + 1])} {Y[t + 1, 2]:.1f}')

    # the Brave-Butters-Kelley series is 12 times the monthly growth rate; divided by 12, its
    # Mariano-Murasawa aggregates must give the quarterly GDP growth of mf_data.csv
    B = pd.read_csv(os.path.join(TDIR, 'bbk_mgdp.csv'), float_precision='round_trip')
    bd = months(B['date'])
    bbk = B['BBKMGDP'].to_numpy() / 12
    four = np.timedelta64(4, 'M')
    tq = np.flatnonzero(~np.isnan(X[:, 5]) & (d >= bd[0] + four) & (d <= bd[-1]))
    w = np.array([1, 2, 3, 2, 1]) / 3
    agg = np.array([w @ bbk[(bd >= d[t] - four) & (bd <= d[t])] for t in tq])
    gap = np.max(np.abs(agg - X[tq, 5]))
    print(f'\nBrave-Butters-Kelley divided by 12, aggregated to {tq.size} quarters: largest gap to '
          f'the quarterly GDP growth {gap:.4f}')
    assert gap < .01, 'the Brave-Butters-Kelley series is not in the units assumed'
    dc, ia, ib = np.intersect1d(d, bd, return_indices=True)
    x, y = gm[ia, 5], bbk[ib]
    k = years(dc) != 2020
    print(f'\nmonthly GDP growth against Brave-Butters-Kelley, {mlab(dc[0])} to {mlab(dc[-1])} '
          f'({dc.size} months): correlation {corr(x, y):.2f}, {corr(x[k], y[k]):.2f} without '
          f'2020; RMS difference {rms(x, y):.2f}, {rms(x[k], y[k]):.2f} without 2020')
    print(f'standard deviations: this model {np.std(x, ddof=1):.2f}, Brave-Butters-Kelley '
          f'{np.std(y, ddof=1):.2f} (without 2020: {np.std(x[k], ddof=1):.2f}, '
          f'{np.std(y[k], ddof=1):.2f})')
    print(f"{'month':<8s} {'model':>8s} {'BBK':>8s}")
    for i in np.flatnonzero((dc >= month('2020-02')) & (dc <= month('2020-07'))):
        print(f'{mlab(dc[i]):<8s} {x[i]:8.2f} {y[i]:8.2f}')

    # ------------------------------------------------------------------
    #  Figures
    # ------------------------------------------------------------------
    lo68, hi68 = res.Y_q[:, :, 1], res.Y_q[:, :, 3]
    dd, bdd = d.astype('datetime64[D]'), bd.astype('datetime64[D]')
    fig, ax = plt.subplots(figsize=(9, 3.8))
    k = d >= month('2006-01')
    kb = bd >= month('2006-01')
    bvt.util.shaded_band(dd[k], lo68[k, 5], hi68[k, 5], ax=ax)
    ax.plot(dd[k], gm[k, 5], 'k', linewidth=1.2)
    ax.plot(bdd[kb], bbk[kb], '--', color=(.85, .33, .1), linewidth=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, linewidth=.5, alpha=.5)
    ax.set_ylabel('percent')
    ax.legend(['68% credible band', 'posterior mean', 'Brave-Butters-Kelley'], loc='lower left',
              frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_gdp.png'), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2))
    k = (d >= month('2019-07')) & (d <= month('2021-06'))
    for j, (ax, title) in enumerate(zip(axes, ['GDP', 'Private fixed investment'])):
        bvt.util.shaded_band(dd[k], lo68[k, iq[j]], hi68[k, iq[j]], ax=ax)
        ax.plot(dd[k], gm[k, iq[j]], 'k', linewidth=1.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, linewidth=.5, alpha=.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('percent')
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_2020.png'), dpi=150)
    plt.close(fig)

    print(f'\nbuild finished in {(time.perf_counter() - t0) / 60:.1f} minutes')
    sys.stdout.file.close()
    sys.stdout = sys.stdout.screen


if __name__ == '__main__':
    main()
