"""check_stability - do the reported quantities move when the chain is rerun?

The build draws 5,000 times after a burn-in of 1,000 at each of 140 origins, and an
inefficiency factor at the last origin measures how correlated those draws are. It does not
establish whether the reported numbers would change under another chain. This script measures
that directly.

At four origins, two ordinary and two around the pandemic, each volatility model is estimated
three times at the build's settings under different seeds, and once at four times the length.
The quantities compared are those the tutorial reports: the joint log predictive likelihood at
one and four quarters, and the predictive probability that GDP growth is negative one quarter
ahead.

This script is not part of the tutorial's results and is not run by the tests. It takes about
seven minutes.

Usage, from the root of the repository:  python tutorials/forecasting/check_stability.py
"""

import os

# one BLAS thread, set before NumPy loads: for the small matrices of these samplers, OpenBLAS's
# threads can be far slower than one (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

p, n0, H = 4, 8, 4
short = (5000, 1000)                 # the build's settings: draws kept, burn-in
long = (20000, 5000)
seeds = [11, 22, 33]

tbl = pd.read_csv(os.path.join(REPO, 'tutorials', 'sv_specification', 'macro5_Q.csv'),
                  float_precision='round_trip')
data = tbl[['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']].to_numpy(dtype=float)
dates = pd.to_datetime(tbl['Date'])
nobs, n = data.shape
k = 1 + n * p
igdp = 4

pick = ['2005-10-01', '2015-10-01', '2019-10-01', '2020-04-01']
pname = ['2005Q4, ordinary', '2015Q4, ordinary', '2019Q4, before the pandemic',
         '2020Q2, in the pandemic']
models = ['VAR-CSV', 'VAR-OISV']

for ic, day in enumerate(pick):
    t = int(np.flatnonzero(dates == day)[0]) + 1          # the rows used
    Y0, Y = data[:n0], data[n0:t]
    cfg = {'ylag': data[t - p:t][::-1].T, 'H': H, 'yobs': data[t:min(t + H, nobs)]}
    print(f'\n=== origin {pname[ic]} ===')
    print(f"{'model':<10s} {'chain':<12s} {'seed':>6s} {'score h=1':>12s} {'score h=4':>12s} "
          f"{'P(GDP<0) h=1':>14s}")
    for im in range(2):
        for (nd, nb), sd in zip([short] * len(seeds) + [long], seeds + seeds[:1]):
            r = bvt.Stream(np.random.default_rng(sd))
            if im == 0:
                D = bvt.models.var_csv(Y0, Y, p, nsim=nd, burnin=nb, rng=r, draws=True).draws
            else:
                D = bvt.models.var_sv(Y0, Y, p, model='OI', nsim=nd, burnin=nb, rng=r,
                                      draws=True).draws
            lj = np.zeros((nd, H))
            mu1 = np.zeros(nd)
            sd1 = np.zeros(nd)
            for d in range(nd):
                if im == 0:
                    dr = {'A': D.A[d].reshape((k, n), order='F'),
                          'Sig': D.Sig[d].reshape((n, n), order='F'), 'h_T': D.h[d, -1],
                          'phi': D.phi[d], 'sigh2': D.sigh2[d]}
                    spec = 'csv'
                else:
                    dr = {'A': D.A[d].reshape((k, n), order='F'),
                          'impact': D.impact[d].reshape((n, n), order='F'), 'h_T': D.h_T[d],
                          'phi': D.phi[d], 'sig2': D.sig2[d]}
                    spec = 'oisv'
                y1, _, j1, s1 = bvt.forecast.simulate(spec, dr, cfg, rng=r)
                lj[d], mu1[d], sd1[d] = j1, y1[0, igdp], s1[0, igdp]
            pneg = np.mean(ndtr((0 - mu1) / sd1))
            print(f'{models[im]:<10s} {f"{nd} draws":<12s} {sd:6d} '
                  f'{bvt.util.logsumexp(lj[:, 0]) - np.log(nd):12.3f} '
                  f'{bvt.util.logsumexp(lj[:, 3]) - np.log(nd):12.3f} {100 * pneg:13.1f}%')

print(f'\nthe three {short[0]}-draw rows of a block differ only in the seed, so their spread is\n'
      f'what another run of the build would change. The {long[0]}-draw row is the reference:\n'
      'a short chain that sits inside the spread of its own seeds and close to it is long\n'
      'enough for the quantity in that column.')
