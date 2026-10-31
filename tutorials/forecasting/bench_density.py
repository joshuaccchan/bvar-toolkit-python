"""bench_density - two ways to score a multi-step density, on the same draws

Given a posterior draw of a VAR with common stochastic volatility, the log predictive
likelihood of a realized value h quarters ahead can be estimated two ways.

  CONDITIONAL GAUSSIAN, which bvt.forecast.simulate uses: simulate only the volatility path,
  then evaluate the realized value against the exact Gaussian that the path implies, with mean
  the VAR iterated forward and variance the sum of Psi_i Sigma_{T+h-i} Psi_i' over i = 0 to
  h-1.

  PATH BASED, written out below as path_score: simulate the data as well, and evaluate the
  realized value against the one-step covariance at the state the simulated path happens to
  reach.

Both are unbiased estimators of the predictive DENSITY, and neither is an unbiased estimator of
its logarithm: by Jensen's inequality the log of an unbiased density estimate sits below the
log density on average, the more so the noisier the estimate. The two are compared here on the
same posterior draws, at an ordinary realized value and at an extreme one, over several
simulation sizes and seeds, to measure that gap.

This script is not part of the tutorial's results and is not run by the tests. It takes about
40 seconds.

Usage, from the root of the repository:  python tutorials/forecasting/bench_density.py
"""

import os

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def path_score(dr, cfg, r):
    """The estimator this tutorial does not use: simulate the data as well as the volatility,
    and score the realized value against the one-step covariance at the state the simulated
    path reaches."""
    k, n = dr['A'].shape
    H = cfg['H']
    lj = np.full(H, np.nan)
    h, phi, sdh = dr['h_T'], dr['phi'], np.sqrt(dr['sigh2'])
    CSig0 = np.linalg.cholesky(dr['Sig'])
    x = np.r_[1, cfg['ylag'].ravel(order='F')]
    for j in range(H):
        h = phi * h + sdh * r.randn()
        CS = np.exp(h / 2) * CSig0
        EY = x @ dr['A']
        if cfg['yobs'].shape[0] > j:
            z = np.linalg.solve(CS, cfg['yobs'][j] - EY)
            lj[j] = -n / 2 * np.log(2 * np.pi) - np.sum(np.log(np.diag(CS))) - .5 * (z @ z)
        x = np.r_[1, EY + CS @ r.randn(n), x[1:-n]]
    return lj


p, n0, H, nsim, burnin = 4, 8, 4, 5000, 1000
sizes = [250, 1000, 5000]
seeds = range(1, 6)

tbl = pd.read_csv(os.path.join(REPO, 'tutorials', 'sv_specification', 'macro5_Q.csv'),
                  float_precision='round_trip')
data = tbl[['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']].to_numpy(dtype=float)
dates = pd.to_datetime(tbl['Date'])
nobs, n = data.shape
k = 1 + n * p

    # an ordinary stretch, and the one that contains the pandemic quarters; t counts the rows used
cases = [int(np.flatnonzero(dates == '2015-10-01')[0]) + 1,
         int(np.flatnonzero(dates == '2019-10-01')[0]) + 1]
cname = ['ordinary (2016Q1-Q4)', 'pandemic (2020Q1-Q4)']

for ic, t in enumerate(cases):
    Y0, Y = data[:n0], data[n0:t]
    cfg = {'ylag': data[t - p:t][::-1].T, 'H': H, 'yobs': data[t:t + H]}
    d0 = dates.iloc[t - 1]
    print(f'\n=== {cname[ic]}, origin {d0.year}Q{(d0.month - 1) // 3 + 1} ===')
    print('realized GDP growth: ' + ' '.join(f'{v:.1f}' for v in cfg['yobs'][:, 4]))

    c = bvt.models.var_csv(Y0, Y, p, nsim=nsim, burnin=burnin, rng=42, draws=True).draws
    draws = [{'A': c.A[d].reshape((k, n), order='F'), 'Sig': c.Sig[d].reshape((n, n), order='F'),
              'h_T': c.h[d, -1], 'phi': c.phi[d], 'sigh2': c.sigh2[d]} for d in range(nsim)]

    for ih in [1, 4]:
        print(f'\n  h = {ih}, joint log score over the five variables')
        print(f"  {'draws':>8s} {'conditional Gaussian':>28s} {'path based':>28s}")
        print(f"  {'':>8s} {'mean':>14s} {'spread':>13s} {'mean':>14s} {'spread':>13s}")
        for ns in sizes:
            sc = np.zeros((len(seeds), 2))
            for js, sd in enumerate(seeds):
                r = bvt.Stream(np.random.default_rng(sd))
                lg = [bvt.forecast.simulate('csv', draws[d], cfg, rng=r)[2][ih - 1]
                      for d in range(ns)]
                r = bvt.Stream(np.random.default_rng(sd))
                lp = [path_score(draws[d], cfg, r)[ih - 1] for d in range(ns)]
                sc[js] = [bvt.util.logsumexp(np.array(lg)) - np.log(ns),
                          bvt.util.logsumexp(np.array(lp)) - np.log(ns)]
            print(f'  {ns:8d} {sc[:, 0].mean():14.2f} {np.ptp(sc[:, 0]):13.2f} '
                  f'{sc[:, 1].mean():14.2f} {np.ptp(sc[:, 1]):13.2f}')

print(f'\nspread is the range over {len(seeds)} seeds, the posterior draws held fixed, so it is\n'
      'simulation noise alone. At h = 1 the two estimators coincide, since no data path\n'
      'has been simulated yet. Beyond that the path-based estimator drifts down and its\n'
      'spread widens, and both effects are worst where the realized value is extreme.')
