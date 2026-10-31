"""build - regenerate the figures and numbers of tutorials/forecasting/README.md

A recursive, pseudo-out-of-sample comparison of three reduced-form BVARs on the five quarterly
series of tutorials/sv_specification: a homoskedastic VAR, a VAR with one common volatility
factor (VAR-CSV), and a VAR with one log-volatility per equation under the order-invariant
impact matrix of Chan, Koop and Yu (2024) (VAR-OISV). None of the three depends on the order of
the columns, so the comparison never asks the reader to justify an ordering.

At each origin every model is re-estimated on the sample to that date and forecast one to four
quarters ahead, one simulated path per posterior draw through bvt.forecast.simulate. Point
forecasts are scored by RMSFE and density forecasts by the log predictive likelihood, which is
the log of the average of exp(log density) over the draws. The homoskedastic model is also
scored exactly, through bvt.forecast.predictive, which measures the simulation noise in the
other two.

Density forecasts are also checked for calibration: how often the realized value fell inside
the interval a model claimed, and how wide that interval was. An interval that is narrow and
covers is worth more than one that is narrow.

Forecasts are grouped by the quarter they are for.

The per-origin scores go to scores_by_origin.npz and the figures are written next to this
file. Everything printed goes to a log in the temporary directory.

Usage, from the root of the repository:  python tutorials/forecasting/build.py
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
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.special import ndtr  # noqa: E402

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


def qlab(d):
    d = pd.Timestamp(d)
    return f'{d.year}Q{(d.month - 1) // 3 + 1}'


def conjugate_draws(Y0, Y, p, nsim, r):
    """Independent draws from the natural conjugate posterior of the homoskedastic VAR: the
    coefficients as nsim x k*n rows stacked column by column, and the covariance matrices."""
    T, n = Y.shape
    k = 1 + n * p
    _, X = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
    A0, VA0, nu0, S0, _ = bvt.priors.niw(p, [.2**2, 100], Y0, Y, 'mlvarsv_ncp')
    KA = X.T @ X
    KA[np.diag_indices(k)] += 1 / VA0
    CKA = np.linalg.cholesky(KA)
    Ahat = solve_triangular(CKA, solve_triangular(CKA, A0 / VA0[:, None] + X.T @ Y, lower=True),
                            lower=True, trans='T')
    Shat = S0 + A0.T @ (A0 / VA0[:, None]) + Y.T @ Y - Ahat.T @ KA @ Ahat
    Shat = (Shat + Shat.T) / 2
    Ad = np.zeros((nsim, k * n))
    Sd = np.zeros((nsim, n, n))
    for d in range(nsim):
        Sig = r.iwishrnd(Shat, nu0 + T)
        A = Ahat + solve_triangular(CKA, r.randn(k, n), lower=True, trans='T') \
            @ np.linalg.cholesky(Sig).T
        Ad[d] = A.ravel(order='F')
        Sd[d] = Sig
    return Ad, Sd


def run_draws(spec, Acol, Scol, D, cfg, n, k, H, r):
    """One simulated path per draw, and the four quantities averaged or collected."""
    nd = Acol.shape[0]
    yh = np.zeros((nd, n, H))
    ld = np.zeros((nd, n, H))
    lj = np.zeros((nd, H))
    sd = np.zeros((nd, n, H))
    for d in range(nd):
        dr = {'A': Acol[d].reshape((k, n), order='F')}
        if spec == 'gauss':
            dr['Sig'] = Scol[d]
        elif spec == 'csv':
            dr.update(Sig=Scol[d].reshape((n, n), order='F'), h_T=D.h[d, -1], phi=D.phi[d],
                      sigh2=D.sigh2[d])
        else:
            dr.update(impact=Scol[d].reshape((n, n), order='F'), h_T=D.h_T[d], phi=D.phi[d],
                      sig2=D.sig2[d])
        y1, l1, j1, s1 = bvt.forecast.simulate(spec, dr, cfg, rng=r)
        yh[d], ld[d], lj[d], sd[d] = y1.T, l1.T, j1, s1.T
    return yh, ld, lj, sd


def store(S, io, im, yh, ld, lj, sd, hs, nobs, t, yobs, r):
    """Average the draws into the point forecast, the log predictive likelihoods, the
    predictive standard deviation, the predictive cdf at the realized value and the widths of
    two intervals, at the two horizons that are evaluated.

    The predictive distribution is the mixture over draws of N(yh, sd^2). Its cdf is the
    average of the component cdfs, which is exact and costs one line. Its quantiles would need
    a root find per interval end, 17,000 of them over this exercise, so the widths come from
    the mixture sampled once per draw, which is accurate enough for a width and thousands of
    times cheaper. The forecasts of forecast_now.py need only a few dozen quantiles, so they
    use bvt.forecast.mixquantile and no sampling."""
    nd = yh.shape[0]
    for ih, h in enumerate(hs):
        if t + h > nobs:
            continue
        y, s = yh[:, :, h - 1], sd[:, :, h - 1]
        S['point'][io, :, ih, im] = y.mean(axis=0)
        S['lpl'][io, :, ih, im] = bvt.util.logsumexp(ld[:, :, h - 1]) - np.log(nd)
        S['ljnt'][io, ih, im] = bvt.util.logsumexp(lj[:, h - 1]) - np.log(nd)
        S['psd'][io, :, ih, im] = np.sqrt(np.mean(s**2, axis=0) + np.var(y, axis=0, ddof=1))
        S['pit'][io, :, ih, im] = np.mean(ndtr((yobs[:, ih] - y) / s), axis=0)
        ys = y + s * r.randn(nd, y.shape[1])
        q = np.quantile(ys, [.100, .900, .025, .975], axis=0, method='hazen')
        S['w80'][io, :, ih, im] = q[1] - q[0]
        S['w95'][io, :, ih, im] = q[3] - q[2]


def main():
    log = os.path.join(tempfile.gettempdir(), 'bvar_toolkit_python_forecasting_build_log.txt')
    sys.stdout = Tee(log)
    print(f'tutorials/forecasting/build.py, {datetime.datetime.now():%Y-%m-%d %H:%M}, '
          f'Python {platform.python_version()}, NumPy {np.__version__}, '
          f'SciPy {scipy.__version__}, bvar_toolkit {bvt.__version__}')
    t_all = time.perf_counter()

    # ---- settings ----
    p, n0, H, hs = 4, 8, 4, [1, 4]
    nsim, burnin = 5000, 1000
    first_forecast = pd.Timestamp('1990-01-01')
    seed0 = 20260921
    mname = ['homoskedastic', 'VAR-CSV', 'VAR-OISV']
    nm = len(mname)

    # ---- data, the panel of tutorials/sv_specification ----
    tbl = pd.read_csv(os.path.join(REPO, 'tutorials', 'sv_specification', 'macro5_Q.csv'),
                      float_precision='round_trip')
    cols = ['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']
    vlabel = ['Unemployment', 'PCE inflation', 'Fed funds', 'NFCI', 'GDP growth']
    data = tbl[cols].to_numpy(dtype=float)
    dates = pd.to_datetime(tbl['Date']).to_numpy()
    nobs, n = data.shape
    k = 1 + n * p
        # every origin has a realized value one quarter ahead; an origin t uses the first t
        # quarters
    origins = np.arange(int(np.flatnonzero(dates >= np.datetime64(first_forecast))[0]), nobs)
    no = origins.size
    print(f'\n{n} variables, {nobs} quarters ({qlab(dates[0])} to {qlab(dates[-1])})')
    print(f'{no} forecast origins, {qlab(dates[origins[0] - 1])} to {qlab(dates[origins[-1] - 1])};'
          f' {nsim} draws after {burnin} burn-in per model and origin')
    print('the models are re-estimated at every origin, on data up to that quarter only')

    S = {key: np.full((no, n, 2, nm), np.nan) for key in ('point', 'lpl', 'psd', 'pit', 'w80',
                                                          'w95')}
    S['ljnt'] = np.full((no, 2, nm), np.nan)
    actual = np.full((no, n, 2), np.nan)
    lpl_exact = np.full((no, n, 2), np.nan)

    for io, t in enumerate(origins):
        Y0, Y = data[:n0], data[n0:t]
        ylag = data[t - p:t][::-1].T
        hmax = min(H, nobs - t)
        cfg = {'ylag': ylag, 'H': H, 'yobs': data[t:t + hmax]}
        for ih, h in enumerate(hs):
            if t + h <= nobs:
                actual[io, :, ih] = data[t + h - 1]

            # ---- model 1: homoskedastic, natural conjugate ----
        r = bvt.Stream(np.random.default_rng(seed0 + io + 1))
        Ad, Sd = conjugate_draws(Y0, Y, p, nsim, r)
        out = run_draws('gauss', Ad, Sd, None, cfg, n, k, H, r)
        store(S, io, 0, *out, hs, nobs, t, actual[io], r)

            # the same model scored exactly, with no simulation
        mu, sdx = bvt.forecast.predictive(Ad, Sd, ylag, H)
        for ih, h in enumerate(hs):
            if t + h > nobs:
                continue
            z = (actual[io, :, ih] - mu[:, :, h - 1]) / sdx[:, :, h - 1]
            ldx = -.5 * np.log(2 * np.pi * sdx[:, :, h - 1]**2) - .5 * z**2
            lpl_exact[io, :, ih] = bvt.util.logsumexp(ldx) - np.log(nsim)

            # ---- model 2: VAR-CSV ----
        r = bvt.Stream(np.random.default_rng(seed0 + io + 1))
        c = bvt.models.var_csv(Y0, Y, p, nsim=nsim, burnin=burnin, rng=r, draws=True)
        out = run_draws('csv', c.draws.A, c.draws.Sig, c.draws, cfg, n, k, H, r)
        store(S, io, 1, *out, hs, nobs, t, actual[io], r)

            # ---- model 3: VAR-SV, order invariant ----
        r = bvt.Stream(np.random.default_rng(seed0 + io + 1))
        s = bvt.models.var_sv(Y0, Y, p, model='OI', nsim=nsim, burnin=burnin, rng=r, draws=True)
        out = run_draws('oisv', s.draws.A, s.draws.impact, s.draws, cfg, n, k, H, r)
        store(S, io, 2, *out, hs, nobs, t, actual[io], r)

        if (io + 1) % 20 == 0:
            print(f'  {io + 1} of {no} origins, {(time.perf_counter() - t_all) / 60:.1f} minutes')
        if io == no - 1:                  # how well the two chains mix at the longest sample
            IFc = bvt.diag.inefficiency_factor(
                np.column_stack([c.draws.phi, c.draws.sigh2, c.draws.Sig[:, 0]]), 200)
            IFs = bvt.diag.inefficiency_factor(
                np.column_stack([s.draws.phi[:, 0], s.draws.sig2[:, 0], s.draws.h_T[:, 0]]), 200)

    # ---- the scores ----
    point, lpl, ljnt, psd, pit = S['point'], S['lpl'], S['ljnt'], S['psd'], S['pit']
    print('\n=== Accuracy relative to the homoskedastic VAR ===')
    odate = dates[origins - 1]                          # the quarter each forecast is made in
    tdate = np.full((no, 2), np.datetime64('NaT'), dtype='datetime64[ns]')
    for ih, h in enumerate(hs):                         # the quarter each forecast is FOR
        ok = origins + h <= nobs
        tdate[ok, ih] = dates[origins[ok] + h - 1]
    covid = np.datetime64('2020-01-01')
    bname = ['all', 'targets through 2019', 'targets 2020 onwards']

    print('forecasts are grouped by the quarter they are FOR, so a four-quarter-ahead\n'
          'forecast made in 2019Q4 counts as a forecast of 2020Q4')
    print(f"\n{'':<16s} {'horizon':>7s} {'group':>22s} {'RMSFE gain':>14s} {'log score gain':>14s}")
    for im in range(1, nm):
        for ih in range(2):
            for ib in range(3):
                sel = [np.ones(no, bool), tdate[:, ih] < covid, tdate[:, ih] >= covid][ib]
                ok = sel & ~np.isnan(actual[:, 0, ih])
                if not ok.any():
                    continue
                e0 = point[ok, :, ih, 0] - actual[ok, :, ih]
                em = point[ok, :, ih, im] - actual[ok, :, ih]
                g = np.median(100 * (1 - np.sqrt(np.mean(em**2, axis=0))
                                     / np.sqrt(np.mean(e0**2, axis=0))))
                dj = np.mean(ljnt[ok, ih, im] - ljnt[ok, ih, 0])
                lbl = mname[im] if ib == 0 else ''
                print(f'{lbl:<16s} {hs[ih]:7d} {bname[ib]:>22s} {g:10.1f}% {dj:14.3f}   '
                      f'({ok.sum()} forecasts)')
    print('\nthe RMSFE column is the median gain across the five variables and the log\n'
          'score column the mean gain in the joint log predictive likelihood, per quarter')

    print('\nby variable, against the homoskedastic VAR')
    for ih in range(2):
        ok = ~np.isnan(actual[:, 0, ih])
        print(f'\n  h = {hs[ih]}, {ok.sum()} forecasts')
        print(f"  {'variable':<22s}" + ''.join(f' {m + " RMSFE":>15s} {m + " score":>15s}'
                                             for m in mname[1:]))
        for i in range(n):
            line = f'  {vlabel[i][:22]:<22s}'
            for im in range(1, nm):
                e0 = point[ok, i, ih, 0] - actual[ok, i, ih]
                em = point[ok, i, ih, im] - actual[ok, i, ih]
                line += (f' {100 * (1 - np.sqrt(np.mean(em**2)) / np.sqrt(np.mean(e0**2))):14.1f}%'
                         f' {np.mean(lpl[ok, i, ih, im] - lpl[ok, i, ih, 0]):15.3f}')
            print(line)
    print("  the score column is the mean gain in that variable's own log predictive\n"
          '  likelihood, so the five do not add up to the joint gain')

    print('\ncalibration: how often the realized value fell inside the interval, and how wide')
    show = [4, 1]                                       # GDP growth and PCE inflation
    for ih in range(2):
        ok = ~np.isnan(actual[:, 0, ih])
        print(f'\n  h = {hs[ih]}, {ok.sum()} forecasts')
        print(f"  {'variable':<16s} {'model':<16s} {'cover 80%':>11s} {'width 80%':>11s} "
              f"{'cover 95%':>11s} {'width 95%':>11s}")
        for i in show:
            for im in range(nm):
                u = pit[ok, i, ih, im]
                print(f'  {vlabel[i][:16]:<16s} {mname[im]:<16s} '
                      f'{100 * np.mean((u > .100) & (u < .900)):10.0f}% '
                      f'{np.mean(S["w80"][ok, i, ih, im]):11.2f} '
                      f'{100 * np.mean((u > .025) & (u < .975)):10.0f}% '
                      f'{np.mean(S["w95"][ok, i, ih, im]):11.2f}')
    print('  a well calibrated 80% interval covers 80% of the time; among those that\n'
          '  do, the narrower one is the more useful')

    print('\nsimulation noise: the homoskedastic model scored by simulation and exactly')
    for ih in range(2):
        ok = ~np.isnan(actual[:, 0, ih])
        d = lpl[ok, :, ih, 0] - lpl_exact[ok, :, ih]
        print(f'  h = {hs[ih]}: mean difference {d.mean():.4f}, largest {np.abs(d).max():.4f}, '
              f'over {ok.sum()} forecasts and {n} variables')

    print(f'\ninefficiency factors at the last origin, {nsim} draws')
    print('  VAR-CSV   phi {:.0f}, sigh2 {:.0f}, Sig(1,1) {:.0f}'.format(*IFc))
    print('  VAR-OISV  phi_1 {:.0f}, sig2_1 {:.0f}, h_T1 {:.0f}'.format(*IFs))

    print('\npredictive standard deviation of GDP growth at h = 1, median over origins')
    for im in range(nm):
        print(f'  {mname[im]:<16s} targets through 2019 '
              f'{np.nanmedian(psd[tdate[:, 0] < covid, 4, 0, im]):6.2f}, targets 2020 onwards '
              f'{np.nanmedian(psd[tdate[:, 0] >= covid, 4, 0, im]):6.2f}')

        # everything the page quotes, per origin, so a question about the split or a single
        # quarter does not need another run
    np.savez_compressed(os.path.join(TDIR, 'scores_by_origin.npz'),
                        odate=odate.astype('datetime64[D]').astype(str),
                        tdate=tdate.astype('datetime64[D]').astype(str), actual=actual,
                        lpl_exact=lpl_exact, mname=mname, hs=hs, vars=cols, vlabel=vlabel,
                        nsim=nsim, burnin=burnin, **S)
    print('\nper-origin scores saved to scores_by_origin.npz')

    # ---- figures ----
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2))
    ptitle = ['all targets', 'targets through 2019']
    colors = [None, (0, .447, .741), (.85, .325, .098)]
    for ib in range(2):
        for ih in range(2):
            ax = axes[ib, ih]
            for im in range(1, nm):
                ok = ~np.isnan(ljnt[:, ih, im])
                if ib == 1:
                    ok &= tdate[:, ih] < covid
                ax.plot(odate[ok], np.cumsum(ljnt[ok, ih, im] - ljnt[ok, ih, 0]), linewidth=1.1,
                        color=colors[im], label=mname[im])
            ax.axhline(0, color='k', linestyle=':', linewidth=.8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'h = {hs[ih]}, {ptitle[ib]}', fontsize=10)
            ax.tick_params(labelsize=8)
            if ih == 0:
                ax.set_ylabel('cumulative log score difference', fontsize=9)
            if ib == 0 and ih == 0:
                ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_cumscore.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for im in range(nm):
        ax.plot(odate, psd[:, 4, 0, im], linewidth=1.1, label=mname[im])
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    ax.set_ylabel('predictive standard deviation')
    ax.set_title('One-quarter-ahead predictive standard deviation of GDP growth', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(TDIR, 'fig_psd.png'), dpi=150)
    plt.close(fig)

    print(f'\nbuild finished in {(time.perf_counter() - t_all) / 60:.1f} minutes')
    sys.stdout.file.close()
    sys.stdout = sys.stdout.screen


if __name__ == '__main__':
    main()
