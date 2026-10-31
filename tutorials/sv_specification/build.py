"""build - regenerate the figures and numbers of tutorials/sv_specification/README.md

The data are the five quarterly series of macro5_Q.csv in this folder: the four of the book's
chapter12/macro4_Q.csv and the NFCI column of its chapter04/GDP_NFCI_merged.csv, joined on the
quarter. The models, priors, lag length and marginal likelihood estimators are those of Chan
(2023), estimated here by ml_varsv.py in this folder. Part 1 estimates the seven
specifications and their log marginal likelihoods, each under two seeds, and saves a summary of
every run in runs/ next to this file; a run whose summary exists with the same settings is not
repeated, so an interrupted build resumes where it stopped. Part 2 reports the marginal
likelihoods, the shrinkage hyperparameters and the outlier probabilities, and Part 3 the MCMC
diagnostics and the repeat estimates. Everything printed goes to a log in the temporary
directory and the figure is written next to this file.

Usage, from the root of the repository:  python tutorials/sv_specification/build.py
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

import bvar_toolkit as bvt  # noqa: E402

TDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TDIR)
import ml_varsv  # noqa: E402


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


def qlab(d):
    d = pd.Timestamp(d)
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


def cached(rdir, i, q, key):
    """The saved run, when its settings match."""
    f = os.path.join(rdir, f'run{i}_{q}.npz')
    if os.path.exists(f):
        with np.load(f, allow_pickle=False) as c:
            if np.array_equal(c['key'], key):
                return {name: c[name] for name in c.files}
    return None


def one_run(sp, Q, nsim, burnin, M, seed, key, rdir, i, q):
    """Estimate one specification and its log marginal likelihood, and save what Parts 2 and
    3 need; the draws of the VAR coefficients and log-volatilities are not kept."""
    t0 = time.perf_counter()
    out = ml_varsv.run_ml(sp['model'], Q, is_kappasym=sp['sym'], nsim=nsim, burnin=burnin,
                          rng=seed, r=sp['r'], M=M)
    name = sp['model'] + (' (symmetric prior)' if sp['sym'] else '') \
        + (f", r = {sp['r']}" if sp['model'] == 'VAR-FSV' else '')
    s = dict(key=key, seed=seed, lml=out.lml, nse=np.nan if out.lmlstd is None else out.lmlstd,
             name=name)
    if 'store_kappa' in out:
        s.update(kappa=np.asarray(out.store_kappa).reshape(nsim, -1), hpara=out.store_hpara,
                 count_phi=out.count_phi)
    if 'CSV_std_mean' in out:
        s['csv_std'] = out.CSV_std_mean
    if 'store_o' in out:
        s.update(pout=np.mean(out.store_o > 1, axis=0), o_mean=out.o_mean, po=out.store_po)
    s['seconds'] = time.perf_counter() - t0
    np.savez(os.path.join(rdir, f'run{i}_{q}.npz'), **s)
    return s


def msd(x):
    return f'{np.mean(x):.3g} ({np.std(x, ddof=1):.2g})'


def nsestr(x):
    return '-' if np.isnan(x) else f'{x:.2f}'


def main():
    log = os.path.join(tempfile.gettempdir(), 'bvar_toolkit_python_sv_specification_build_log.txt')
    sys.stdout = Tee(log)
    print(f'tutorials/sv_specification/build.py, {datetime.datetime.now():%Y-%m-%d %H:%M}, '
          f'Python {platform.python_version()}, NumPy {np.__version__}, '
          f'SciPy {scipy.__version__}, bvar_toolkit {bvt.__version__}')
    t0 = time.perf_counter()

        # the five series as the book uses them: growth rates and inflation annualized, the
        # two rates and the NFCI in levels, no further scaling. The order is the normalization
        # of the Cholesky and factor specifications: unemployment leads, so the first factor
        # is normalized on a real variable, inflation follows for a nominal one, and GDP
        # growth, whose pandemic quarters are the most extreme, comes last
    cols = ['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']
    raw = pd.read_csv(os.path.join(TDIR, 'macro5_Q.csv'), float_precision='round_trip')
    Q = raw[cols].to_numpy(dtype=float)
    qd = pd.to_datetime(raw['Date']).to_numpy()
    assert not np.isnan(Q).any(), 'the panel has missing values'
    gap = np.diff(qd).astype('timedelta64[D]').astype(int)
    assert np.all((gap >= 89) & (gap <= 93)), 'the quarters are not consecutive'
    n0, n = 8, len(cols)
    T = Q.shape[0] - n0
    print(f'data: {Q.shape[0]} quarters, {qlab(qd[0])} to {qlab(qd[-1])}')
    print(f'estimation sample {qlab(qd[n0])} to {qlab(qd[-1])}, T = {T} after {n0} initial '
          f'conditions, n = {n}, p = 4')

    # ------------------------------------------------------------------
    #  Part 1. The seven specifications
    # ------------------------------------------------------------------
    nsim, burnin, M = 20000, 1000, 10000                # Section 5 of Chan (2023)
    spec = [dict(model='VAR-NCP', sym=False, r=1), dict(model='VAR-CSV', sym=False, r=1),
            dict(model='VAR-SV', sym=False, r=1), dict(model='VAR-SV', sym=True, r=1),
            dict(model='VAR-FSV', sym=False, r=1), dict(model='VAR-FSV', sym=False, r=2),
            dict(model='VAR-SVO', sym=False, r=1)]
    nrun = len(spec)
    seeds = [20260919, 8177]                # the repeat run refits chain, density and weights
    key = np.array([nsim, burnin, M, Q.sum(), 0])     # a saved run is reused only under these
    rdir = os.path.join(TDIR, 'runs')
    os.makedirs(rdir, exist_ok=True)
    S = [cached(rdir, i + 1, 1, key) for i in range(nrun)]
    R = [cached(rdir, i + 1, 2, key) for i in range(nrun)]
    todo = [i for i in range(nrun) if S[i] is None or R[i] is None]
    print(f'\n=== Part 1: {nsim} draws after {burnin} burn-in, {M} importance-sampling draws, '
          'two seeds ===')
    print(f'{nrun - len(todo)} of {nrun} specifications saved by an earlier build')
    for i in todo:
        if S[i] is None:
            S[i] = one_run(spec[i], Q, nsim, burnin, M, seeds[0] + i + 1, key, rdir, i + 1, 1)
        if R[i] is None:
            R[i] = one_run(spec[i], Q, nsim, burnin, M, seeds[1] + i + 1, key, rdir, i + 1, 2)
        print(f"run {i + 1}, {S[i]['name']}: {S[i]['seconds'] / 60:.1f} and "
              f"{R[i]['seconds'] / 60:.1f} minutes")
    print('run times in minutes, both seeds:' + ''.join(
        f" {i + 1} {(float(S[i]['seconds']) + float(R[i]['seconds'])) / 60:.1f};"
        for i in range(nrun)))

    # ------------------------------------------------------------------
    #  Part 2. Marginal likelihoods, shrinkage and outliers
    # ------------------------------------------------------------------
    print('\n=== Part 2: log marginal likelihoods ===')
    lml = np.array([float(s['lml']) for s in S])
    nse = np.array([float(s['nse']) for s in S])
    names = [str(s['name']) for s in S]
    tab1 = [0, 1, 2, 4, 5, 6]                           # Table 1; run 4 is the symmetric prior
    ib = tab1[int(np.argmax(lml[tab1]))]
    print('\nTable 1')
    print(f'| Model | log marginal likelihood | numerical standard error | difference from '
          f'{names[ib]} |')
    print('|---|---|---|---|')
    for i in tab1:
        print(f'| {names[i]} | {lml[i]:.1f} | {nsestr(nse[i])} | {lml[i] - lml[ib]:.1f} |')
    print('VAR-NCP has no standard error: its log marginal likelihood is available in closed form')

    print('\nTable 2: the prior (posterior mean and standard deviation of each shrinkage '
          'hyperparameter)')
    print('| Model | log marginal likelihood | own lags | other lags | impact matrix |')
    print('|---|---|---|---|---|')
    for i in [1, 3, 2, 4, 5, 6]:
        k = S[i]['kappa']
        c = [''] * 3
        if k.shape[1] == 1:                             # VAR-CSV: one kappa for all lags
            c[0] = c[1] = msd(k[:, 0])
        else:
            c[0], c[1] = msd(k[:, 0]), msd(k[:, 1])
            if k.shape[1] == 3:
                c[2] = msd(k[:, 2])
        print(f'| {names[i]} | {lml[i]:.1f} | {c[0]} | {c[1]} | {c[2]} |')
    print(f'symmetric minus asymmetric prior, VAR-SV: {lml[3] - lml[2]:.1f}; VAR-SV (symmetric) '
          f'minus VAR-CSV: {lml[3] - lml[1]:.1f}')

    print('\n=== Part 2b: the outlier component of VAR-SVO ===')
    so = S[6]
    dq = qd[n0:]
    print(f"posterior mean of the outlier probability po: {np.mean(so['po']):.4f} "
          f"(sd {np.std(so['po'], ddof=1):.4f}); prior mean {2.5 / 40:.4f}")
    print(f"expected number of outlier quarters: {np.sum(so['pout']):.1f} of {T}")
    print('quarters with an outlier probability above 0.5:')
    big = np.flatnonzero(so['pout'] > 0.5)
    for t in big:
        print(f"  {qlab(dq[t])}: probability {so['pout'][t]:.2f}, posterior mean of o_t "
              f"{so['o_mean'][t]:.2f}")
    cs = S[1]['csv_std']
    print(f'VAR-CSV: posterior mean of exp(h_t/2) from {cs.min():.2f} ({qlab(dq[np.argmin(cs)])}) '
          f'to {cs.max():.2f} ({qlab(dq[np.argmax(cs)])})')

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.7))
    axes[0].plot(dq, cs, 'k-', linewidth=1.2)
    axes[0].set_ylabel('$e^{h_t/2}$')
    axes[0].set_title('Common standard-deviation multiplier, VAR-CSV (posterior mean)', fontsize=10)
    axes[1].bar(dq, so['pout'], width=np.timedelta64(92, 'D'), color=(.4, .4, .4),
                edgecolor='none')
    axes[1].set_ylim(0, 1.2)
    axes[1].set_ylabel('probability')
    if big.size:
        axes[1].text(dq[big[-1]], 1.12, ', '.join(qlab(dq[t]) for t in big) + ' ',
                     ha='right', fontsize=9)
    axes[1].set_title('Posterior outlier probability under VAR-SVO', fontsize=10)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_outliers.png'), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    #  Part 3. MCMC diagnostics
    # ------------------------------------------------------------------
    print('\n=== Part 3: MCMC diagnostics (bvt.diag) ===')
    L = 200
    print(f'inefficiency factors at L = {L} by parameter group, and the largest gap between the')
    print('posterior means of the two runs over all hyperparameters, in posterior sd')
    print('| Model | kappa | mu | phi | sigma^2 | largest gap between runs |')
    print('|---|---|---|---|---|---|')
    gname = ['kappa', 'mu', 'phi', 'sigma^2']
    drift = [[], [], [], []]               # |late - early| mean, in posterior sd, by group
    for i in [1, 2, 3, 4, 5, 6]:
        s, r = S[i], R[i]
        nk = s['kappa'].shape[1]
        m = s['hpara'].shape[1]
        if m == 2:                                      # VAR-CSV: [phi sig2], no mu
            grp = [s['kappa'], None, s['hpara'][:, :1], s['hpara'][:, 1:]]
            gid = np.r_[np.zeros(nk), 2, 3]
        else:
            q = m // 3
            grp = [s['kappa'], s['hpara'][:, :q], s['hpara'][:, q:2 * q], s['hpara'][:, 2 * q:]]
            gid = np.r_[np.zeros(nk), np.ones(q), 2 * np.ones(q), 3 * np.ones(q)]
        cells = []
        for g in grp:
            if g is None:
                cells.append('-')
                continue
            IF = bvt.diag.inefficiency_factor(g, L)
            cells.append(f'{IF.min():.0f} to {IF.max():.0f}')
        X1 = np.hstack([s['kappa'], s['hpara']])
        X2 = np.hstack([r['kappa'], r['hpara']])
        sd = np.vstack([X1, X2]).std(axis=0, ddof=1)
        print(f"| {names[i]} | {' | '.join(cells)} | "
              f'{np.max(np.abs(X1.mean(axis=0) - X2.mean(axis=0)) / sd):.2f} |')
        N = X1.shape[0]
        e = slice(0, int(np.floor(.1 * N + .5)))
        late = slice(int(np.floor(.5 * N + .5)), N)
        d = np.abs(np.vstack([(X1[late].mean(axis=0) - X1[e].mean(axis=0)) / sd,
                              (X2[late].mean(axis=0) - X2[e].mean(axis=0)) / sd]))
        for g in range(4):
            drift[g] += list(d[:, gid == g].ravel(order='F'))
    print('\ndrift within a run, between the first tenth and the last half, in posterior sd,')
    print('over both runs of the six specifications')
    for g in range(4):
        print(f'  {gname[g]:<8s} median {np.median(drift[g]):.2f}, largest {np.max(drift[g]):.2f}')
    print('numerical standard errors of the log marginal likelihoods: '
          f'{np.nanmin(nse[1:]):.2f} to {np.nanmax(nse[1:]):.2f}')

    print('\nrepeat runs: each specification estimated again from a second seed, which refits the')
    print('chain, the importance density and the weights')
    print('| Model | log ML, seed 1 | log ML, seed 2 | difference | standard error |')
    print('|---|---|---|---|---|')
    for i in range(nrun):
        print(f"| {names[i]} | {float(S[i]['lml']):.1f} | {float(R[i]['lml']):.1f} | "
              f"{float(R[i]['lml']) - float(S[i]['lml']):.1f} | {nsestr(nse[i])} |")
        # what the normalization costs when the most extreme series leads: the same one-factor
        # model and seeds, with GDP growth moved to the front
    alt = [4, 0, 1, 2, 3]
    keyalt = np.r_[key[:4], 1]
    A = []
    for q in range(2):
        a = cached(rdir, nrun + 1, q + 1, keyalt)
        if a is None:
            a = one_run(spec[4], Q[:, alt], nsim, burnin, M, seeds[q] + 5, keyalt, rdir,
                        nrun + 1, q + 1)
        A.append(a)
    print(f"\nVAR-FSV, r = 1, with {cols[alt[0]]} first instead of {cols[0]}: log ML "
          f"{float(A[0]['lml']):.1f} and {float(A[1]['lml']):.1f} from the two seeds,")
    print(f"{abs(float(A[1]['lml']) - float(A[0]['lml'])):.1f} apart, against "
          f"{abs(float(R[4]['lml']) - float(S[4]['lml'])):.1f} in the order above")

    hours = sum(float(x['seconds']) for x in S + R + A) / 3600
    print(f'\nbuild finished in {(time.perf_counter() - t0) / 60:.1f} minutes (runs included: '
          f'{hours:.1f} hours)')
    sys.stdout.file.close()
    sys.stdout = sys.stdout.screen


if __name__ == '__main__':
    main()
