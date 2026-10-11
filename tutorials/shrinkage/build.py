"""build - regenerate the figures and numbers of tutorials/shrinkage/README.md

The data and settings are those of the working-paper version of Chan (2022), whose
replication package supplies data/chan2019wp_acp: 21 quarterly US variables, four lags,
and the prior set on the structural-form coefficients. Part 1 evaluates the closed-form
marginal likelihood under the three priors of the paper's Table 2 and over the grid of its
Figure 1, and checks the optima against Table 2 and the package's own run. The figure it
writes uses a logarithmic grid instead, which spans the three priors. Part 2 draws from the
posterior under each prior and scans the lag length. Part 3 repeats the paper's recursive
forecasting exercise, whose code is not in the package. The figures are written next to
this file, and everything printed also goes to a log in the temporary directory.

Usage, from the root of the repository:  python tutorials/shrinkage/build.py
"""

import datetime
import os
import platform
import sys
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

TDIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TDIR))


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


def qlab(i):
    """The quarter of data row i (from 0), the first row being 1959Q3."""
    return f'{1959 + (2 + i) // 4}Q{(2 + i) % 4 + 1}'


def stars(z):
    """Significance stars for a two-sided test at the 10, 5 and 1% levels."""
    return '*' * sum(int(abs(z) > c) for c in (1.645, 1.96, 2.576))


def logmeanexp(x):
    """log(mean(exp(x))) down each column, without overflow."""
    m = np.max(x, axis=0)
    return m + np.log(np.mean(np.exp(x - m), axis=0))


def dm_stat(d, h):
    """Diebold-Mariano statistic for the loss differential d, with a Newey-West long-run
    variance using h-1 lags."""
    T = d.size
    u = d - np.mean(d)
    lrv = (u @ u) / T
    for lag in range(1, h):
        lrv = lrv + 2 * (1 - lag / h) * (u[lag:] @ u[:-lag]) / T
    return np.mean(d) / np.sqrt(lrv / T)


def main():
    log = os.path.join(tempfile.gettempdir(), 'bvar_toolkit_python_shrinkage_build_log.txt')
    sys.stdout = Tee(log)
    print(f"tutorials/shrinkage/build.py, {datetime.datetime.now():%Y-%m-%d %H:%M}, "
          f"Python {platform.python_version()}, bvar_toolkit {bvt.__version__}")
    t0 = time.perf_counter()

        # the data as main_BVAR_ACP.m of the package reads them
    raw = np.loadtxt(os.path.join(REPO, 'data', 'chan2019wp_acp', 'macrodata_Q_2018Q4.csv'),
                     delimiter=',')
    var_id = np.array([1, 2, 18, 22, 34, 35, 57, 59, 76, 81, 95, 97, 120, 123, 133, 138, 145,
                       148, 152, 160, 245]) - 1
    data = raw[:238, var_id]
    vnames = ['GDP', 'Consumption', 'Disposable income', 'Industrial production',
              'Capacity utilization', 'Payroll employment', 'Civilian employment',
              'Unemployment rate', 'Hours', 'Housing starts', 'PCE prices', 'GDP deflator',
              'CPI', 'PPI', 'Real earnings', 'Productivity', '3-month T-bill', '10-year yield',
              'Baa spread', 'Real M1', 'S&P 500']
    assert qlab(int(np.argmax(data[:, 7]))) == '1982Q4', 'the unemployment rate should peak in 1982Q4'
    p, n0 = 4, 8
    Y0, Y = data[:n0], data[n0:]
    T, n = Y.shape
    k = n * p + 1
    _, Z = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
    sig2 = bvt.priors.resid_var_ar4(Y0, Y)
    k_subj = np.array([.04, .0016, 1, 100])
    labs = ['symmetric', 'subjective', 'asymmetric']
    print(f'data: {qlab(n0)} to {qlab(data.shape[0] - 1)}, T = {T} quarters after {n0} initial '
          f'conditions, n = {n}, p = {p}')

    # ------------------------------------------------------------------
    # Part 1. The paper's Table 2 and Figure 1
    # ------------------------------------------------------------------
    print('\n=== Part 1: the marginal likelihood of the 21-variable VAR ===')
    ml_asym, k_asym = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, [.04, .04], 'stru')
    ml_sym, k_sym = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, None, 'stru', symmetric=True)
    ml_subj = bvt.ml.acp(p, Y, Z, bvt.priors.acp_stru(n, p, k_subj, sig2))

        # the package's own run prints kappa to three decimals and the log marginal
        # likelihood to integers; Table 2 of the paper adds the subjective prior
    assert (round(k_sym[0], 3) == 0.039 and round(k_asym[0], 3) == 0.406
            and round(k_asym[1], 3) == 0.009 and round(ml_sym) == -9436
            and round(ml_asym) == -9201), 'the optima differ from the package run'
    assert round(ml_subj) == -9372, 'the subjective prior does not reproduce Table 2 of the paper'
    print("optima equal the package's run; Table 2 of the paper reproduced")

    print('\nTable 1')
    print('| Prior | kappa2 (own lags) | kappa3 (other lags) | log marginal likelihood | difference |')
    print('|---|---|---|---|---|')
    K = [k_sym, k_subj, k_asym]
    ML = [ml_sym, ml_subj, ml_asym]
    for j in range(3):
        print(f'| {labs[j]} | {K[j][0]:.4f} | {K[j][1]:.4f} | {ML[j]:.1f} | {ML[j] - ml_asym:.1f} |')
    print(f'full precision: asymmetric ({k_asym[0]:.10g}, {k_asym[1]:.10g}) {ml_asym:.6f}; '
          f'symmetric {k_sym[0]:.10g} {ml_sym:.6f}; subjective {ml_subj:.6f}')
    print(f'kappa2 / kappa3: {k_asym[0] / k_asym[1]:.1f}; kappa2 / symmetric kappa: '
          f'{k_asym[0] / k_sym[0]:.1f}; kappa3 / symmetric kappa: {k_asym[1] / k_sym[0]:.2f}')

        # the grid of the paper's Figure 1 (main_BVAR_ACP.m line 58); under a flat prior on
        # (kappa2, kappa3) the surface is proportional to their joint posterior density, and
        # the summaries below are grid approximations
    g2 = np.linspace(.25, .65, 41)
    g3 = np.linspace(.002, .02, 91)
    K2, K3 = np.meshgrid(g2, g3)
    lml = np.array([bvt.ml.acp(p, Y, Z, bvt.priors.acp_stru(n, p, [a, b, 1, 100], sig2))
                    for a, b in zip(K2.ravel(), K3.ravel())]).reshape(K2.shape)
    im = np.unravel_index(np.argmax(lml), lml.shape)
    print(f'grid of {K2.shape[0]} x {K2.shape[1]} over kappa2 in [{g2[0]:.2f}, {g2[-1]:.2f}] and '
          f'kappa3 in [{g3[0]:.3f}, {g3[-1]:.3f}]: maximum {lml[im]:.3f} at ({K2[im]:.2f}, {K3[im]:.4f})')
    w = np.exp(lml - lml.max())
    w = w / w.sum()
    m2, m3 = w.sum(axis=0), w.sum(axis=1)

    def q(g, m, a):
        return g[np.argmax(np.cumsum(m) >= a)]
    print(f'grid approximations on that support: kappa2 mean {g2 @ m2:.3f}, 90% interval '
          f'[{q(g2, m2, .05):.2f}, {q(g2, m2, .95):.2f}]; kappa3 mean {g3 @ m3:.4f}, 90% interval '
          f'[{q(g3, m3, .05):.4f}, {q(g3, m3, .95):.4f}]')
    print(f'mass at the edges of the grid: kappa2 {m2[0] + m2[-1]:.2g}, kappa3 {m3[0] + m3[-1]:.2g}')

        # the figure spans all three priors, which a linear grid around the optimum leaves
        # out, so it uses a logarithmic one
    f2 = np.logspace(np.log10(min(k_asym[0] / 10, k_sym[0], k_subj[0]) / 1.5), np.log10(k_asym[0] * 4), 61)
    f3 = np.logspace(np.log10(min(k_asym[1] / 10, k_subj[1]) / 1.5),
                     np.log10(max(k_asym[1] * 10, k_sym[1]) * 1.5), 61)
    F2, F3 = np.meshgrid(f2, f3)
    lf = np.array([bvt.ml.acp(p, Y, Z, bvt.priors.acp_stru(n, p, [a, b, 1, 100], sig2))
                   for a, b in zip(F2.ravel(), F3.ravel())]).reshape(F2.shape)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.contour(F2, F3, np.exp(lf - lf.max()), levels=np.arange(1, 10) / 10, linewidths=1,
               cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    dlo, dhi = max(f2[0], f3[0]), min(f2[-1], f3[-1])
    ax.plot([dlo, dhi], [dlo, dhi], 'k--', linewidth=0.8)
    ax.text(dhi, dhi, r' $\kappa_2 = \kappa_3$', va='top', fontsize=10)
    kmark = [k_asym[:2], k_sym[:2], k_subj[:2]]
    style = [dict(marker='*', mfc='w', ms=14), dict(marker='o', mfc='k', ms=6), dict(marker='s', mfc='k', ms=6)]
    off = [1.9, 1.15, 1.15]                         # the asymmetric label clears its contours
    for j, lab in enumerate(['asymmetric', 'symmetric', 'subjective']):
        ax.plot(kmark[j][0], kmark[j][1], color='k', linestyle='none', **style[j])
        ax.text(kmark[j][0] * off[j], kmark[j][1], lab, fontsize=10, va='center')
    ax.set_xlim(f2[0], f2[-1])
    ax.set_ylim(f3[0], f3[-1])
    ax.set_xlabel(r'$\kappa_2$ (own lags), smaller is tighter')
    ax.set_ylabel(r'$\kappa_3$ (other lags), smaller is tighter')
    ax.set_title('Marginal likelihood relative to its maximum')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_contour.png'), dpi=150)
    plt.close(fig)

        # the same comparison in the 15-variable application of the published version, with
        # the settings of the package's main_ACP_apps.m (lines 16, 37, 38, 40 and 53)
    d22 = pd.read_excel(os.path.join(REPO, 'data', 'chan2022_qe_acp', 'database_2019Q4.xlsx'),
                        header=None).iloc[2:, 1:].to_numpy(dtype=float)
    p22, n_init = 5, 8
    idx_ns22 = np.array([1, 2, 4, 5, 10, 11, 12, 13, 15]) - 1
    Y0q, Yq = d22[:n_init], d22[n_init:]
    _, Zq = bvt.util.build_lags(np.vstack([Y0q[-p22:], Yq]), p22)
    lq_a, _ = bvt.priors.acp_opt_kappa(Y0q, Yq, Zq, p22, [.04, .0016], 'redu', idx_ns22)
    lq_s, _ = bvt.priors.acp_opt_kappa(Y0q, Yq, Zq, p22, None, 'redu', idx_ns22, symmetric=True)
    print(f'published version, 15 variables: log ML asymmetric {lq_a:.1f}, symmetric {lq_s:.1f}, '
          f'difference {lq_a - lq_s:.1f}')

    # ------------------------------------------------------------------
    # Part 2. What the choice changes: coefficients and lag length
    # ------------------------------------------------------------------
    print('\n=== Part 2: posterior means of the VAR coefficients, and the lag length ===')
    nsim = 10000
    Abar = []
    for j in range(3):
        Alp, Beta, Sg = bvt.samplers.acp_theta_sig(Y0, Y, p, bvt.priors.acp_stru(n, p, K[j], sig2), nsim,
                                                   rng=20260919)
        Bt, _ = bvt.structural.reduced_form(Alp, Beta, Sg)
        Abar.append(Bt.mean(axis=0).reshape(n, k).T)
    own = np.zeros((n * p, n), dtype=bool)
    for lag in range(p):
        own[lag * n:(lag + 1) * n] = np.eye(n, dtype=bool)
    oth = ~own
        # coefficient of y_{r,t-l} in equation i, times s_r/s_i; all prior means are zero
    s = np.sqrt(sig2)
    scl = np.tile(s, p)[:, None] / s[None, :]
    first = np.zeros((n * p, n), dtype=bool)
    first[:n] = True
    print(f'{nsim} posterior draws under each prior; mean absolute posterior mean of the lag')
    print('coefficients in standard-deviation units')
    print(f'| Prior | own lags | other lags | other lags, lag 1 | other lags, lags 2-{p} |')
    print('|---|---|---|---|---|')
    Ld = []
    for j in range(3):
        Ld.append(Abar[j][1:] * scl)
        print(f'| {labs[j]} | {np.mean(np.abs(Ld[j][own])):.4f} | {np.mean(np.abs(Ld[j][oth])):.4f} | '
              f'{np.mean(np.abs(Ld[j][oth & first])):.4f} | {np.mean(np.abs(Ld[j][oth & ~first])):.4f} |')
    r_own = [np.mean(np.abs(Ld[j][own])) for j in range(3)]
    r_oth = [np.mean(np.abs(Ld[j][oth])) for j in range(3)]
    print(f'own lags: asymmetric / symmetric {r_own[2] / r_own[0]:.2f}, subjective / symmetric {r_own[1] / r_own[0]:.2f}')
    print(f'other lags: asymmetric / symmetric {r_oth[2] / r_oth[0]:.2f}, subjective / symmetric {r_oth[1] / r_oth[0]:.2f}')

    pmax = n0
    lag_tab = np.zeros((pmax, 5))
    for pp in range(1, pmax + 1):
        _, Zp = bvt.util.build_lags(np.vstack([Y0[-pp:], Y]), pp)
        la, ka = bvt.priors.acp_opt_kappa(Y0, Y, Zp, pp, [.04, .04], 'stru')
        ls, ks = bvt.priors.acp_opt_kappa(Y0, Y, Zp, pp, None, 'stru', symmetric=True)
        lag_tab[pp - 1] = [ka[0], ka[1], la, ks[0], ls]
    assert lag_tab[p - 1, 2] == ml_asym, f'the lag scan at p = {p} does not reproduce Part 1'
    print('\nTable 2')
    print('| p | kappa2 | kappa3 | log ML, asymmetric | kappa, symmetric | log ML, symmetric |')
    print('|---|---|---|---|---|---|')
    for pp in range(pmax):
        r = lag_tab[pp]
        print(f'| {pp + 1} | {r[0]:.3f} | {r[1]:.4f} | {r[2]:.1f} | {r[3]:.3f} | {r[4]:.1f} |')
    print(f'lag length with the highest log marginal likelihood: asymmetric p = '
          f'{np.argmax(lag_tab[:, 2]) + 1}, symmetric p = {np.argmax(lag_tab[:, 4]) + 1}')
    print('asymmetric minus symmetric, by lag length: '
          + ' '.join(f'{d:.1f}' for d in lag_tab[:, 2] - lag_tab[:, 4]))

    # ------------------------------------------------------------------
    # Part 3. Recursive forecasts, 1985Q1 to the end of the sample
    # ------------------------------------------------------------------
    print('\n=== Part 3: recursive forecasts ===')
    print('pseudo-out-of-sample: the exercise truncates this one vintage at each origin, so the\n'
          'early samples hold the revised values, and each forecast conditions on the hyperparameters\n'
          'that maximize the marginal likelihood at that origin')
    hs = [1, 4]
    H = max(hs)
    origins = np.arange(101, data.shape[0] - 1)    # 1984Q4 to 2018Q3
    assert qlab(origins[0]) == '1984Q4'
    nor, nsim_f = origins.size, 10000
    point = np.full((nor, n, 2, 3), np.nan)
    lpl = np.full((nor, n, 2, 3), np.nan)
    actual = np.full((nor, n, 2), np.nan)
    jlpl = np.full((nor, 3), np.nan)
    kpath = np.full((nor, 3), np.nan)
    print(f'{nor} forecast origins, {qlab(origins[0])} to {qlab(origins[-1])}; {nsim_f} posterior draws '
          f'per prior and origin; p = {p} throughout')
    for io, t in enumerate(origins):
        Yf, Yf1 = data[n0:t + 1], data[n0:t + 2]
        _, Zf = bvt.util.build_lags(np.vstack([Y0[-p:], Yf]), p)
        _, Zf1 = bvt.util.build_lags(np.vstack([Y0[-p:], Yf1]), p)
        s2f = bvt.priors.resid_var_ar4(Y0, Yf)
        _, ka = bvt.priors.acp_opt_kappa(Y0, Yf, Zf, p, [.04, .04], 'stru')
        _, ks = bvt.priors.acp_opt_kappa(Y0, Yf, Zf, p, None, 'stru', symmetric=True)
        kpath[io] = [ka[0], ka[1], ks[0]]
        Kf = [ks, k_subj, ka]
        ylag = data[t - p + 1:t + 1][::-1].T
        for ih, h in enumerate(hs):
            if t + h < data.shape[0]:
                actual[io, :, ih] = data[t + h]
        for j in range(3):
            prior = bvt.priors.acp_stru(n, p, Kf[j], s2f)
                # all 21 variables one quarter ahead, exactly: with the prior known at t held
                # fixed, p(y_t+1 | y_1:t) = p(y_1:t+1) / p(y_1:t)
            jlpl[io, j] = bvt.ml.acp(p, Yf1, Zf1, prior) - bvt.ml.acp(p, Yf, Zf, prior)
                # common random numbers across the priors
            Alp, Beta, Sg = bvt.samplers.acp_theta_sig(Y0, Yf, p, prior, nsim_f, rng=20260919 + int(t))
            Bt, St = bvt.structural.reduced_form(Alp, Beta, Sg)
            mu, sd = bvt.forecast.predictive(Bt, St, ylag, H)
            for ih, h in enumerate(hs):
                if np.isnan(actual[io, 0, ih]):
                    continue
                m, sv = mu[:, :, h - 1], sd[:, :, h - 1]
                point[io, :, ih, j] = m.mean(axis=0)
                lpl[io, :, ih, j] = logmeanexp(-0.5 * np.log(2 * np.pi) - np.log(sv)
                                               - 0.5 * ((actual[io, :, ih] - m) / sv) ** 2)
    print(f'optimal kappa over the origins: kappa2 {kpath[:, 0].min():.3f} to {kpath[:, 0].max():.3f}, '
          f'kappa3 {kpath[:, 1].min():.4f} to {kpath[:, 1].max():.4f}, symmetric kappa '
          f'{kpath[:, 2].min():.3f} to {kpath[:, 2].max():.3f}')
    print(f'number of forecasts: h = 1: {np.sum(~np.isnan(actual[:, 0, 0]))}, '
          f'h = 4: {np.sum(~np.isnan(actual[:, 0, 1]))}')

        # gains of the asymmetric prior as in the paper: 100(1 - RMSFE ratio) and
        # 100 x (ALPL difference); DM positive when the asymmetric prior is more accurate
    gR, gA, zR, zA = (np.zeros((n, 2, 2)) for _ in range(4))
    for v in range(n):
        for ih, h in enumerate(hs):
            ok = ~np.isnan(actual[:, v, ih])
            e = point[ok, v, ih, :] - actual[ok, v, ih][:, None]
            ls = lpl[ok, v, ih, :]
            r = np.sqrt(np.mean(e ** 2, axis=0))
            a = np.mean(ls, axis=0)
            for b in range(2):                      # benchmark: 0 symmetric, 1 subjective
                gR[v, ih, b] = 100 * (1 - r[2] / r[b])
                gA[v, ih, b] = 100 * (a[2] - a[b])
                zR[v, ih, b] = dm_stat(e[:, b] ** 2 - e[:, 2] ** 2, h)
                zA[v, ih, b] = dm_stat(ls[:, 2] - ls[:, b], h)
    bn = ['symmetric', 'subjective']
    print('\nTable 3 (gains of the asymmetric prior: RMSFE in percent, ALPL as 100 x the')
    print('difference in average log predictive likelihood; significant = two-sided DM test at 5%)')
    print('| Benchmark | h | median RMSFE gain | RMSFE gains / significant / significant losses | '
          'median 100 x ALPL difference | ALPL gains / significant / significant losses |')
    print('|---|---|---|---|---|---|')
    for b in range(2):
        for ih, h in enumerate(hs):
            print(f'| {bn[b]} | {h} | {np.median(gR[:, ih, b]):.2f} | {np.sum(gR[:, ih, b] > 0)} / '
                  f'{np.sum(zR[:, ih, b] > 1.96)} / {np.sum(zR[:, ih, b] < -1.96)} | '
                  f'{np.median(gA[:, ih, b]):.2f} | {np.sum(gA[:, ih, b] > 0)} / '
                  f'{np.sum(zA[:, ih, b] > 1.96)} / {np.sum(zA[:, ih, b] < -1.96)} |')
    print('\nTable 4 (gains of the asymmetric prior by variable: RMSFE in percent, ALPL as 100 x the')
    print('difference; stars: two-sided DM test at 10/5/1%)')
    print('| Variable | RMSFE h=1, vs symmetric | RMSFE h=4, vs symmetric | ALPL h=1, vs symmetric | '
          'ALPL h=4, vs symmetric | RMSFE h=1, vs subjective | RMSFE h=4, vs subjective | '
          'ALPL h=1, vs subjective | ALPL h=4, vs subjective |')
    print('|---|---|---|---|---|---|---|---|---|')
    for v in range(n):
        cells = []
        for b in range(2):
            for G, Zs in ((gR, zR), (gA, zA)):
                for ih in range(2):
                    cells.append(f'{G[v, ih, b]:.1f}{stars(Zs[v, ih, b])}')
        print(f'| {vnames[v]} | ' + ' | '.join(cells) + ' |')
    tot = jlpl.sum(axis=0)
    print(f'\njoint log predictive likelihood of all {n} variables one quarter ahead, exact (a difference')
    print(f'of log marginal likelihoods), summed over {nor} origins: symmetric {tot[0]:.1f}, '
          f'subjective {tot[1]:.1f}, asymmetric {tot[2]:.1f}')
    print(f'DM, asymmetric against: symmetric {dm_stat(jlpl[:, 2] - jlpl[:, 0], 1):.2f}, '
          f'subjective {dm_stat(jlpl[:, 2] - jlpl[:, 1], 1):.2f}')

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    ttl = [['RMSFE gain against the symmetric prior', 'RMSFE gain against the subjective prior'],
           ['ALPL gain against the symmetric prior', 'ALPL gain against the subjective prior']]
    x = np.arange(n)
    for mtr, G in enumerate((gR, gA)):
        for b in range(2):
            ax = axes[mtr, b]
            ax.bar(x - 0.225, G[:, 0, b], 0.45, color=(0, 0.447, 0.741), label='one quarter ahead')
            ax.bar(x + 0.225, G[:, 1, b], 0.45, color=(0.85, 0.325, 0.098), label='four quarters ahead')
            ax.axhline(0, color='k', linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(vnames, rotation=60, ha='right', fontsize=8)
            ax.set_xlim(-0.6, n - 0.4)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_title(ttl[mtr][b], fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            if mtr == 0 and b == 0:
                ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_forecasts.png'), dpi=150)
    plt.close(fig)

    print(f'\nbuild finished in {(time.perf_counter() - t0) / 60:.1f} minutes')
    sys.stdout.file.close()
    sys.stdout = sys.stdout.screen


if __name__ == '__main__':
    main()
