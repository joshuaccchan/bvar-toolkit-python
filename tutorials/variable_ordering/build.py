"""build - regenerate the figures and numbers of tutorials/variable_ordering/README.md

Part 1 estimates a four-variable VAR with stochastic volatility, under the Cholesky and the
order-invariant specifications, in the published and the reversed order of the variables, and
once more under a second seed, at the published chain length of 30,000 draws after 5,000
burn-in. Part 2 reads the 20-variable posterior means that the Chan, Koop and Yu (2024)
package ships in its results_mat folder. Part 3 recomputes the forecast comparison of the
package's Table3_forecasting.m from the forecasts it ships and checks the result against the
run of that script captured in data/chan_koop_yu2024_jbes_oisv. The eight files of
results_mat, 32 MB, are downloaded on first use from the archive of the package in the MATLAB
toolkit and checked against their md5 sums. Everything printed goes to a log in the temporary
directory and the figures are written next to this file.

Usage, from the root of the repository:  python tutorials/variable_ordering/build.py
"""

import datetime
import hashlib
import itertools
import os
import platform
import re
import sys
import tempfile
import time
import urllib.request

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

TDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TDIR))
PKG = os.path.join(REPO, 'data', 'chan_koop_yu2024_jbes_oisv')
URL = ('https://raw.githubusercontent.com/joshuaccchan/bvar-toolkit/refs/tags/as-published/'
       'chan_koop_yu2024_jbes_oisv/replications/chan_koop_yu2024_jbes_oisv/legacy/results_mat/')
RESULTS = {'fullsampleCS1.mat': '3b71687cca4cd12e1fa3ad226ace5c56',
           'fullsampleCS2.mat': '2dd90ce2c956ed36431aa261de534ee6',
           'fullsampleOI1.mat': '518619326423ee3ba1179ee5841a4c8d',
           'fullsampleOI2.mat': '9f7314a625ad7f2345a765baa1e4b518',
           'forecastingCS1-cluster.mat': '7676716aead97c46071bc83f507a9a34',
           'forecastingCS2-cluster.mat': '396b544c351772fdcf2c736910e28fec',
           'forecastingOI1-cluster.mat': '2e99542ea90b790785e93c4d3645e73c',
           'forecastingOI2-cluster.mat': 'aeea89c75d80778fb1ac87e970e3108a'}

STYLE = [('-', (0, .447, .741), 1.0),          # Cholesky SV, published order
         ('--', (.85, .325, .098), 1.0),       # Cholesky SV, reversed order
         ('-', (.62, .62, .62), 2.6),          # order-invariant SV, published order
         (':', (0, 0, 0), 1.4)]                # order-invariant SV, reversed order
LEGEND = ['Cholesky SV, published order', 'Cholesky SV, reversed order',
          'order-invariant SV, published order', 'order-invariant SV, reversed order']


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


def results_file(name):
    """A file of the package's results_mat folder, downloaded once and checked by md5."""
    path = os.path.join(PKG, 'results_mat', name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'downloading {name}')
        urllib.request.urlretrieve(URL + name, path + '.part')
        os.replace(path + '.part', path)
    with open(path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    assert md5 == RESULTS[name], f'{name} has md5 {md5}, not {RESULTS[name]}; delete it and rerun'
    return path


def relgap(a, b):
    """The mean absolute difference of two paths, as a percentage of their mean level."""
    return 100 * np.mean(np.abs(a - b)) / np.mean(np.abs(a) + np.abs(b)) * 2


def cpath(S, i, j):
    """The correlation path of variables i and j of a T x n x n covariance path."""
    return S[:, i, j] / np.sqrt(S[:, i, i] * S[:, j, j])


def months(start, T):
    """T months from start, as days for plotting."""
    m0 = np.datetime64(start, 'M')
    return np.arange(m0, m0 + np.timedelta64(T, 'M')).astype('datetime64[D]')


def draw_four(ax, x, Y, labels=True):
    """The four paths in the columns of Y (Cholesky 1 and 2, order-invariant 1 and 2),
    order-invariant first so the thinner Cholesky lines stay visible on top."""
    h = [None] * 4
    for q in [2, 3, 1, 0]:
        h[q], = ax.plot(x, Y[:, q], STYLE[q][0], color=STYLE[q][1], linewidth=STYLE[q][2])
    ax.spines[['top', 'right']].set_visible(False)
    if labels:
        ax.legend(h, LEGEND, loc='upper right', fontsize=8, frameon=False)


def nw_hac(z, nlag):
    """The Newey-West long-run variance of a series, as NWhac in EvalFore.m."""
    T = z.size
    s = z @ z / T
    for i in range(1, nlag + 1):
        g = z[i:] @ z[:-i] / T
        s += (1 - i / (nlag + 1)) * 2 * g
    return s


def dm_pvalue(e_bench, e_alt):
    """The two-sided p-value of the Diebold-Mariano test of EvalFore.m, which compares the
    squares of the two columns, with the Newey-West variance at round(12 (T/100)^(1/4))
    lags."""
    T = e_bench.size
    maxlag = int(np.floor(12 * (T / 100)**.25 + .5))
    dif2 = e_alt**2 - e_bench**2
    dm = np.sqrt(T) * np.mean(dif2) / np.sqrt(nw_hac(dif2, maxlag))
    return 2 * (1 - ndtr(abs(dm)))


def table3():
    """Table3_forecasting.m of the package: RMSFE and average log predictive likelihood of
    the four models for the first four variables at horizons 1, 6 and 12, with the p-values
    of Diebold-Mariano tests against the first model turned into star counts."""
    n = 20
    yh = []
    for m in ['CS1', 'CS2', 'OI1', 'OI2']:
        y = scipy.io.loadmat(results_file(f'forecasting{m}-cluster.mat'))['yhatmedian']
        yh.append([y[:, :, 0], y[:-5, :, 5], y[:-11, :, 11]])
    re_, dm_ = [], []
    for i in range(4):
        cols = []
        for kind in ('R', 'L'):
            for ih, start in enumerate([11, 6, 0]):
                X = np.zeros((yh[0][2].shape[0], 4))
                for m in range(4):
                    y = yh[m][ih][start:]
                    rev = m in (1, 3)                    # the reversed models list the
                    j = n - 1 - i if rev else i          # variables backwards
                    X[:, m] = y[:, j] - y[:, n + j] if kind == 'R' else y[:, 2 * n + j]
                stat = np.sqrt(np.mean(X**2, axis=0)) if kind == 'R' else X.mean(axis=0)
                pv = np.array([np.nan] + [dm_pvalue(X[:, 0], X[:, m]) for m in range(1, 4)])
                cols.append((stat, pv))
        re_.append(np.column_stack([c[0] for c in cols]))
        dm_.append(np.column_stack([c[1] for c in cols]))
    re_, dm_ = np.vstack(re_), np.vstack(dm_)
    stars = np.select([dm_ <= .01, dm_ <= .05, dm_ <= .10], [3, 2, 1], 0)
    return re_, stars


def golden_table():
    """The numbers and star counts of the table in the captured MATLAB run, row by row."""
    text = open(os.path.join(PKG, 'golden_log_Table3_forecasting.txt'), encoding='utf-8').read()
    text = text.split('Table_re_withDM =')[1].split('Elapsed')[0]
    cells = re.findall(r"\{'([^']*)'\s*\}", text)
    nums = [float(re.search(r'-?\d+\.\d{3}(?!\d)', c).group()) for c in cells]
    stars = [len(re.search(r'\^\{(\**)\}', c).group(1)) if '^' in c else 0 for c in cells]
    return np.array(nums).reshape(16, 6), np.array(stars).reshape(16, 6)


def main():
    log = os.path.join(tempfile.gettempdir(), 'bvar_toolkit_python_variable_ordering_build_log.txt')
    sys.stdout = Tee(log)
    print(f'tutorials/variable_ordering/build.py, {datetime.datetime.now():%Y-%m-%d %H:%M}, '
          f'Python {platform.python_version()}, NumPy {np.__version__}, '
          f'SciPy {scipy.__version__}, bvar_toolkit {bvt.__version__}')
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    #  Part 1. Four variables at the published chain length
    # ------------------------------------------------------------------
    data = np.loadtxt(os.path.join(PKG, 'FRED_MD_20vars.csv'), delimiter=',')
    var_id = [3, 5, 11, 12]                        # IP, unemployment, PCE inflation, fed funds
    vnames = ['IP', 'unemployment', 'PCE inflation', 'fed funds']
    p, n0, nsim, burnin = 13, 24, 30000, 5000      # the package's settings
    seeds = [20260915, 31415926]
    cfg = [('OI', False, seeds[0]), ('OI', True, seeds[0]), ('OI', False, seeds[1]),
           ('CS', False, seeds[0]), ('CS', True, seeds[0]), ('CS', False, seeds[1])]
    res = []
    for model, rev, seed in cfg:
        order = var_id[::-1] if rev else var_id
        t1 = time.perf_counter()
        r = bvt.models.var_sv(data[:n0, order], data[n0:, order], p, model=model, nsim=nsim,
                              burnin=burnin, rng=seed)
        if rev:                                    # back to the published order
            r.Sig_mean = r.Sig_mean[:, ::-1, ::-1]
        res.append(r)
        print(f'{model}, {"reversed" if rev else "published"} order, seed {seed}: '
              f'{(time.perf_counter() - t1) / 60:.1f} minutes')
    n = len(var_id)
    prs = list(itertools.combinations(range(n), 2))
    pairs = [(3, 4), (3, 5), (0, 1), (0, 2)]       # CS orderings, CS seeds, OI orderings, OI seeds
    gap = np.array([[np.mean(np.abs(cpath(res[a].Sig_mean, i, j) - cpath(res[b].Sig_mean, i, j)))
                     for a, b in pairs] for i, j in prs])

    print(f'\n=== Part 1: four variables, {nsim} draws after {burnin} burn-in ===')
    print('| | Cholesky SV, two orderings | Cholesky SV, two seeds | order-invariant SV, two '
          'orderings | order-invariant SV, two seeds |')
    print('|---|---|---|---|---|')
    for q in range(n):
        print(f'| var({vnames[q]}) |' + ''.join(
            f' {relgap(res[a].Sig_mean[:, q, q], res[b].Sig_mean[:, q, q]):.1f}% |'
            for a, b in pairs))
    for (i, j), g in zip(prs, gap):
        print(f'| corr({vnames[i]}, {vnames[j]}) |' + ''.join(f' {x:.3f} |' for x in g))

    i, j = prs[int(np.argmax(gap[:, 0]))]          # the pair with the largest Cholesky gap
    order = [3, 4, 0, 1]                           # CS 1, CS 2, OI 1, OI 2
    T4 = res[0].Sig_mean.shape[0]
    v4 = np.column_stack([res[q].Sig_mean[:, 3, 3] for q in order])
    c4 = np.column_stack([cpath(res[q].Sig_mean, i, j) for q in order])
    print(f'range over time of the {vnames[i]} x {vnames[j]} correlation:')
    for q in range(4):
        print(f'  {LEGEND[q]:<36s} {c4[:, q].min():.2f} to {c4[:, q].max():.2f}')
    x = months('1961-03', T4)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.1))
    draw_four(axes[0], x, v4)
    axes[0].set_title('Variance of the federal funds rate equation', fontsize=10)
    draw_four(axes[1], x, c4, labels=False)
    axes[1].set_title(f'Correlation between the {vnames[i]} and {vnames[j]} equations',
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_4var.png'), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    #  Part 2. The 20-variable VAR: the posterior means the package ships
    # ------------------------------------------------------------------
    names = ['real personal income', 'real PCE', 'real M&T sales', 'IP', 'capacity utilization',
             'unemployment', 'payrolls', 'hours', 'hourly earnings', 'PPI finished goods',
             'PPI commodities', 'PCE inflation', 'fed funds', 'housing starts', 'S&P 500',
             'USD/GBP', '1-year spread', '10-year spread', 'BAA spread', 'ISM new orders']
    order20 = [3, 5, 11, 12, 0, 1, 2, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19]
    lab = [names[q] for q in order20]              # labels in the published order
    S = []
    for m in ['CS1', 'CS2', 'OI1', 'OI2']:
        Sm = scipy.io.loadmat(results_file(f'fullsample{m}.mat'))['Sig_mean']
        if m.endswith('2'):                        # reversed order: back to the published one
            Sm = Sm[:, ::-1, ::-1]
        S.append(Sm)
    T20, n20 = S[0].shape[:2]
    vg = np.array([[relgap(S[0][:, q, q], S[1][:, q, q]), relgap(S[2][:, q, q], S[3][:, q, q])]
                   for q in range(n20)])
    prs20 = list(itertools.combinations(range(n20), 2))
    cg = np.array([[np.mean(np.abs(cpath(S[0], a, b) - cpath(S[1], a, b))),
                    np.mean(np.abs(cpath(S[2], a, b) - cpath(S[3], a, b)))] for a, b in prs20])

    print(f'\n=== Part 2: 20 variables, shipped posterior means (T = {T20}, n = {n20}) ===')
    print('| | Cholesky SV | order-invariant SV |\n|---|---|---|')
    print('| Variances, median change | {:.1f}% | {:.1f}% |'.format(*np.median(vg, axis=0)))
    print('| Variances, largest change | {:.1f}% | {:.1f}% |'.format(*vg.max(axis=0)))
    print('| Correlations, median change | {:.3f} | {:.3f} |'.format(*np.median(cg, axis=0)))
    print('| Correlations, 90th percentile | {:.3f} | {:.3f} |'.format(
        *np.percentile(cg, 90, axis=0, method='hazen')))
    print('| Correlations, largest change | {:.3f} | {:.3f} |'.format(*cg.max(axis=0)))
    print('variances where the Cholesky change exceeds the largest order-invariant change: '
          f'{np.sum(vg[:, 0] > vg[:, 1].max())} of {n20}')
    print('correlations where the Cholesky change exceeds the largest order-invariant change: '
          f'{np.sum(cg[:, 0] > cg[:, 1].max())} of {len(prs20)}')
    print('largest variance changes (Cholesky / order-invariant):')
    for q in np.argsort(-vg[:, 0], kind='stable')[:5]:
        print(f'  {lab[q]:<22s} {vg[q, 0]:6.1f}%  {vg[q, 1]:5.1f}%')
    o = np.argsort(-cg[:, 0], kind='stable')
    print('largest correlation changes (Cholesky / order-invariant):')
    for q in o[:5]:
        a, b = prs20[q]
        print(f'  {lab[a]:<22s} x {lab[b]:<22s} {cg[q, 0]:.3f}  {cg[q, 1]:.3f}')
    top = prs20[o[0]]
    print('the four variables of Part 1 inside the 20-variable VAR (Cholesky / order-invariant):')
    for q, (a, b) in enumerate(prs20):
        if b <= 3:
            print(f'  corr({lab[a]}, {lab[b]}): {cg[q, 0]:.3f}  {cg[q, 1]:.3f}')
    for q in range(4):
        print(f'  var({lab[q]}): {vg[q, 0]:.1f}%  {vg[q, 1]:.1f}%')

    v20 = np.column_stack([S[q][:, 3, 3] for q in range(4)])
    c20 = np.column_stack([cpath(S[q], *top) for q in range(4)])
    print(f'time average of the {lab[top[0]]} x {lab[top[1]]} correlation (Cholesky 1, 2, '
          'order-invariant 1, 2): ' + ' '.join(f'{v:.2f}' for v in c20.mean(axis=0)))
    print('its range over time under order-invariant SV, published order: '
          f'{c20[:, 2].min():.2f} to {c20[:, 2].max():.2f}')
    print('peak of the fed funds variance (Cholesky 1, 2, order-invariant 1, 2): '
          + ' '.join(f'{v:.2f}' for v in v20.max(axis=0)))
    x = months('1961-03', T20)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.1))
    draw_four(axes[0], x, v20)
    axes[0].set_title('Variance of the federal funds rate equation, 20-variable VAR', fontsize=10)
    draw_four(axes[1], x, c20, labels=False)
    axes[1].set_title(f'Correlation between the {lab[top[0]]} and {lab[top[1]]} equations',
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_20var_paths.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 3.15))
    rank = np.arange(1, o.size + 1)
    ax.plot(rank, cg[o, 0], 'o', color=STYLE[0][1], markersize=3, label='Cholesky SV')
    ax.plot(rank, cg[o, 1], 'o', color='k', markersize=3, label='order-invariant SV')
    ax.set_xlabel(f'the {o.size} correlations, ranked by the change under Cholesky SV')
    ax.set_ylabel('mean absolute change')
    ax.set_xlim(0, o.size + 1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_20var_gaps.png'), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    #  Part 3. The forecast comparison, recomputed from the shipped forecasts
    # ------------------------------------------------------------------
    fc, stars = table3()
    g_num, g_star = golden_table()
    assert np.array_equal(np.array([float(f'{v:.3f}') for v in fc.ravel()]).reshape(16, 6),
                          g_num), 'the recomputed forecast table differs from the MATLAB run'
    assert np.array_equal(stars, g_star), 'the significance stars differ from the MATLAB run'
    print(f'\nforecast table recomputed from the shipped forecasts: matches the captured MATLAB '
          f'run ({g_num.size} numbers and their stars)')
    vars4 = ['IP', 'unemployment', 'PCE inflation', 'fed funds']
    print('\n=== Part 3: forecasts, 1970:03 onward ===')
    print('| Variable | Model | RMSFE h=1 | RMSFE h=6 | RMSFE h=12 | ALPL h=1 | ALPL h=6 | '
          'ALPL h=12 |')
    print('|---|---|---|---|---|---|---|---|')
    for v in range(4):
        for m in range(4):
            rw = 4 * v + m
            cells = [(f'{fc[rw, c]:.4g}' if c < 3 else f'{fc[rw, c]:.3f}') + '*' * stars[rw, c]
                     for c in range(6)]
            print(f'| {vars4[v]} | {LEGEND[m]} | ' + ' | '.join(cells) + ' |')
    print('RMSFE: lower is better. ALPL: average log predictive likelihood, higher is better. '
          'Stars: two-sided Diebold-Mariano test against Cholesky SV in the published order, '
          '10/5/1%.')
    blocks = [fc[4 * v:4 * v + 4, :3] for v in range(4)]
    spread = max(np.max(100 * (b.max(axis=0) - b.min(axis=0)) / b.min(axis=0)) for b in blocks)
    print(f'largest spread of RMSFE across the four models, by variable and horizon: {spread:.1f}%')

    def pair(a, b):
        return 100 * np.abs(fc[a, :3] - fc[b, :3]) / np.minimum(fc[a, :3], fc[b, :3])
    cs = max(pair(4 * v, 4 * v + 1).max() for v in range(4))
    oi = max(pair(4 * v + 2, 4 * v + 3).max() for v in range(4))
    print(f'largest RMSFE difference between the two orderings: Cholesky SV {cs:.1f}%, '
          f'order-invariant SV {oi:.1f}%')

    print(f'\nbuild finished in {(time.perf_counter() - t0) / 60:.1f} minutes')
    sys.stdout.file.close()
    sys.stdout = sys.stdout.screen


if __name__ == '__main__':
    main()
