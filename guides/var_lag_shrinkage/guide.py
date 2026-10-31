"""guide - the code of guides/var_lag_shrinkage/README.md: the lag length and the shrinkage of the
BVAR of guides/var_forecast chosen by the marginal likelihood at the end of each quarter from 120
to 239 of the sample of guides/var_simulation, against 24 alternatives: a fixed lag length from
one to eight, with c1 chosen by the marginal likelihood, c1 = 0.2^2 or c1 = 10, a nearly flat
prior. The one-quarter-ahead forecasts are scored by the joint log predictive likelihood, with
all the data up to each origin and with its last 80 quarters. Then the checks, and forecasts
averaged over c1. Section 2 and the functions at the end are the code to adapt.

Reads guides/data/macro3_Q.csv. Saves fig_choice.png next to this file.

Usage, from the root of the repository:  python guides/var_lag_shrinkage/guide.py
"""

import os

# one BLAS thread, set before NumPy loads (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from scipy.stats import gamma  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def mat2str(x, spec='g'):
    """a row of numbers as MATLAB's mat2str prints it, each in the format spec"""
    return '[' + ' '.join(f'{v:{spec}}' for v in np.ravel(x)) + ']'


def main():
    # ---- 1. the true model of guides/var_simulation, and the sample of 240 quarters simulated there ----
    D = pd.read_csv(os.path.join(os.path.dirname(HERE), 'data', 'macro3_Q.csv'),
                    float_precision='round_trip')
    Yd, Zd = bvt.util.build_lags(D.iloc[:, 1:4].to_numpy(), 4)
    Atrue = np.linalg.lstsq(Zd, Yd, rcond=None)[0]
    Ud = Yd - Zd @ Atrue
    Sigtrue = Ud.T @ Ud / (Yd.shape[0] - Zd.shape[1])
    rng = np.random.default_rng(2)
    data, _ = bvt.util.simulate_var(Atrue, Sigtrue, 240, rng=rng)   # GDP growth, inflation, unemployment
    Tall, n = data.shape

    # ---- 2. the choices at each origin: the code to adapt ----
    pmax = 8
    c1fix = [.2**2, 10]                               # c1 = 0.2^2, and a nearly flat prior
    origins = np.arange(120, Tall)                    # the number of quarters up to each origin
    no = origins.size
    LS = np.full((no, pmax, 3, 2), np.nan)            # joint log scores: p, the c1, the window
    SE = np.full((no, n, pmax, 3, 2), np.nan)         # squared forecast errors
    pstar = np.zeros((no, 2), dtype=int)              # the choices of the marginal likelihood
    c1star = np.zeros((no, 2))
    for w in range(2):
        for j in range(no):
            t = origins[j]
            W = data[:t] if w == 0 else data[t - 80:t]   # all data, or the last 80
            LS[j, :, :, w], SE[j, :, :, :, w], pstar[j, w], c1star[j, w] = \
                choose_and_forecast(W, data[t], pmax, c1fix)

    # ---- 3. the scores ----
    als = LS.mean(axis=0)                             # average joint log scores, pmax x 3 x 2
    rmsfe = np.sqrt(SE.mean(axis=0))                  # n x pmax x 3 x 2
    lsml = np.zeros((no, 2))                          # the scores of the choices of the ML
    for w in range(2):
        lsml[:, w] = LS[np.arange(no), pstar[:, w] - 1, 0, w]
    lst = np.zeros(no)                                # the scores of the true model
    for j in range(no):
        t = origins[j]
        x = np.concatenate([[1.0], data[t - 4:t][::-1].ravel()])
        lst[j] = mvt_logpdf(data[t], x @ Atrue, Sigtrue, np.inf)

    # ---- the results ----
    wn = ['all data up to the origin', 'the last 80 quarters']
    for w in range(2):
        print(f'\n{wn[w]}: average joint log scores by lag length')
        print('  p                 ' + ''.join(f'{p:8d}' for p in range(1, pmax + 1)))
        print('  c1 by the ML      ' + ''.join(f'{v:8.3f}' for v in als[:, 0, w]))
        print('  c1 = 0.2^2        ' + ''.join(f'{v:8.3f}' for v in als[:, 1, w]))
        print('  c1 = 10           ' + ''.join(f'{v:8.3f}' for v in als[:, 2, w]))
        ib = np.unravel_index(np.argmax(als[:, :, w]), (pmax, 3))
        best = als[ib[0], ib[1], w]
        print(f'  the choice of the marginal likelihood {lsml[:, w].mean():.3f}; the best '
              f'alternative {best:.3f} (p = {ib[0] + 1}, c1 {ib[1]})')
        print(f'  chosen lag lengths: '
              f'{mat2str(np.histogram(pstar[:, w], np.arange(.5, pmax + 1))[0], "d")} '
              f'(counts of p = 1 to {pmax})')
        print(f'  chosen c1: median {np.median(c1star[:, w]):.3f}, from {c1star[:, w].min():.3f} '
              f'to {c1star[:, w].max():.3f}')
        for p in [4, 8]:
            print(f'  RMSFE, p = {p}: c1 by the ML {mat2str(rmsfe[:, p - 1, 0, w], ".3g")}, '
                  f'c1 = 0.2^2 {mat2str(rmsfe[:, p - 1, 1, w], ".3g")}, '
                  f'c1 = 10 {mat2str(rmsfe[:, p - 1, 2, w], ".3g")}')
    print(f'\nthe true model: average joint log score {lst.mean():.3f}')
    print('last 80 quarters, eight lags: RMSFE with c1 = 10 over that with c1 chosen '
          f'{mat2str(rmsfe[:, 7, 2, 1] / rmsfe[:, 7, 0, 1], ".3g")}')

    # ---- 4. the checks ----
    # the log marginal likelihood is the sum of the one-step log predictive densities of the
    # observations, each from the posterior given the observations before it
    W = data[:origins[0]]
    p = 4
    s2 = bvt.priors.resid_var_ar4(W[pmax - 4:pmax], W[pmax:])
    Y, Z = bvt.util.build_lags(W[pmax - p:], p)
    VA = np.concatenate([[10**2], np.kron(1 / np.arange(1, p + 1)**2, c1star[0, 0] / s2)])
    T = Y.shape[0]
    S0 = np.diag(s2)
    nu0 = n + 2
    lsum = 0
    for t in range(T):
        if t == 0:                    # the prior predictive of the first observation
            Ahat, KA, Shat = np.zeros((Z.shape[1], n)), np.diag(1 / VA), S0
        else:
            _, Ahat, KA, Shat = ncp_lml(Y[:t], Z[:t], VA, nu0, S0)
        df = nu0 + t - n + 1
        x = Z[t]
        lsum += mvt_logpdf(Y[t], x @ Ahat, (1 + x @ np.linalg.solve(KA, x)) * Shat / df, df)
    lml = ncp_lml(Y, Z, VA, nu0, S0)[0]
    gapml = lsum - lml
    print(f'\nfirst origin, p = 4: log ML {lml:.6f}, sum of {T} one-step log predictive densities '
          f'{lsum:.6f}, gap {gapml:.1e}')
    assert abs(gapml) < 1e-8, \
        'the marginal likelihood is not the product of the one-step predictive densities'

    # ---- 5. forecasts averaged over c1: the adaptation ----
    lmode, lsd = .2, .4                               # lambda = sqrt(c1): gamma, mode 0.2, sd 0.4
    th = (-lmode + np.sqrt(lmode**2 + 4 * lsd**2)) / 2
    a = lmode / th + 1                                # its scale and shape
    grid = np.exp(np.linspace(np.log(1e-3), np.log(10), 81))   # c1, evenly spaced in log c1
    # the prior density of log c1
    lpri = gamma.logpdf(np.sqrt(grid), a, scale=th) + np.log(np.sqrt(grid) / 2)
    lsav = np.zeros((no, 2))
    for w in range(2):
        for j in range(no):
            t = origins[j]
            W = data[:t] if w == 0 else data[t - 80:t]
            lsav[j, w] = average_over_c1(W, data[t], 4, pmax, grid, lpri)
        print(f'{wn[w]}, p = 4: average joint log score with c1 chosen {als[3, 0, w]:.4f}, '
              f'averaged over c1 {lsav[:, w].mean():.4f}, '
              f'difference {lsav[:, w].mean() - als[3, 0, w]:.4f}')

    # ---- the figure ----
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    allv = np.append(als.ravel(), lst.mean())
    yl = (allv.min() - .03, allv.max() + .03)
    lags = np.arange(1, pmax + 1)
    for w, ax in enumerate(axes):
        hm, = ax.plot(lags, als[:, 0, w], 'k-o', lw=1, ms=4, mfc='none')
        hs, = ax.plot(lags, als[:, 1, w], 'k--', lw=.75)
        hf, = ax.plot(lags, als[:, 2, w], 'k:', lw=1.2)
        ht = ax.axhline(lst.mean(), color='0.6', lw=.75)
        ax.set_xlim(1, pmax)
        ax.set_ylim(yl)
        ax.set_xlabel('lag length', fontsize=10)
        ax.set_title('Estimated on ' + wn[w], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.legend([hm, hs, hf, ht], ['$c_1$ chosen by the marginal likelihood', '$c_1 = 0.2^2$',
                                  '$c_1 = 10$', 'True model'],
               loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.savefig(os.path.join(HERE, 'fig_choice.png'), dpi=150)
    plt.close(fig)


def choose_and_forecast(W, y1, pmax, c1fix):
    """for lag lengths 1 to pmax, the log marginal likelihood maximized over c1, and the joint log
    scores and the squared errors of the one-step forecasts of the outcome y1, with c1 chosen and
    with each value in c1fix, all estimated on the window W; pstar and c1star are the choices of
    the marginal likelihood"""
    n = W.shape[1]
    ns = 1 + len(c1fix)
    ls = np.zeros((pmax, ns))
    se = np.zeros((n, pmax, ns))
    lml = np.zeros(pmax)
    c1 = np.zeros(pmax)
    s2 = bvt.priors.resid_var_ar4(W[pmax - 4:pmax], W[pmax:])   # AR(4) variances
    for p in range(1, pmax + 1):
        Y, Z = bvt.util.build_lags(W[pmax - p:], p)            # the same Y for every p
        VA = lambda c: np.concatenate([[10**2], np.kron(1 / np.arange(1, p + 1)**2, c / s2)])
        res = minimize_scalar(lambda lc: -ncp_lml(Y, Z, VA(np.exp(lc)), n + 2, np.diag(s2))[0],
                              bounds=(np.log(1e-3), np.log(10)), method='bounded')
        lml[p - 1] = -res.fun
        c1[p - 1] = np.exp(res.x)
        x = np.concatenate([[1.0], W[::-1][:p].ravel()])       # x_{t+1}
        cs = [c1[p - 1], *c1fix]
        for s in range(ns):
            _, Ahat, KA, Shat = ncp_lml(Y, Z, VA(cs[s]), n + 2, np.diag(s2))
            df = Y.shape[0] + 3                                # nuhat - n + 1
            m = x @ Ahat
            Psi = (1 + x @ np.linalg.solve(KA, x)) * Shat / df   # the Student-t forecast
            ls[p - 1, s] = mvt_logpdf(y1, m, Psi, df)
            se[:, p - 1, s] = (y1 - m)**2
    pstar = int(np.argmax(lml)) + 1                   # the lag length with the largest ML
    c1star = c1[pstar - 1]
    return ls, se, pstar, c1star


def average_over_c1(W, y1, p, pmax, grid, lpri):
    """the joint log score of the one-step forecast of y1 from a VAR(p) estimated on W, with the
    forecasts for the values of c1 in grid averaged by their posterior probabilities under the
    prior density lpri of log c1"""
    n = W.shape[1]
    s2 = bvt.priors.resid_var_ar4(W[pmax - 4:pmax], W[pmax:])
    Y, Z = bvt.util.build_lags(W[pmax - p:], p)
    x = np.concatenate([[1.0], W[::-1][:p].ravel()])
    lw = np.zeros(grid.size)
    lp = np.zeros(grid.size)
    for g in range(grid.size):
        VA = np.concatenate([[10**2], np.kron(1 / np.arange(1, p + 1)**2, grid[g] / s2)])
        lw[g], Ahat, KA, Shat = ncp_lml(Y, Z, VA, n + 2, np.diag(s2))
        df = Y.shape[0] + 3
        lp[g] = mvt_logpdf(y1, x @ Ahat, (1 + x @ np.linalg.solve(KA, x)) * Shat / df, df)
    lw = lw + lpri
    lw = lw - bvt.util.logsumexp(lw)                  # the posterior probabilities of the grid
    ls = bvt.util.logsumexp(lw + lp)                  # the log of the averaged predictive density
    return ls


def ncp_lml(Y, Z, VA, nu0, S0):
    """the log marginal likelihood under vec(A) | Sig ~ N(0, Sig kron diag(VA)) and
    Sig ~ IW(nu0, S0), with the posterior quantities Ahat, KA and Shat"""
    T, n = Y.shape
    KA = np.diag(1 / VA) + Z.T @ Z
    Ahat = np.linalg.solve(KA, Z.T @ Y)
    E = Y - Z @ Ahat                                  # the residuals at the posterior mean
    # S0 + Y'Y - Ahat' KA Ahat, without the cancellation
    Shat = S0 + E.T @ E + Ahat.T @ (Ahat / VA[:, None])
    Shat = (Shat + Shat.T) / 2
    lml = (-n * T / 2 * np.log(np.pi) - n / 2 * (np.sum(np.log(VA)) + bvt.util.ldet(KA))
           + nu0 / 2 * bvt.util.ldet(S0) - (nu0 + T) / 2 * bvt.util.ldet(Shat)
           + bvt.util.mgammaln(n, (nu0 + T) / 2) - bvt.util.mgammaln(n, nu0 / 2))
    return lml, Ahat, KA, Shat


def mvt_logpdf(y, m, Psi, df):
    """the log density at y of the multivariate Student-t with location m, scale matrix Psi and df
    degrees of freedom; df = np.inf gives the Gaussian"""
    n = y.size
    C = np.linalg.cholesky(Psi)
    z = solve_triangular(C, y - m, lower=True)
    if np.isinf(df):
        lp = -n / 2 * np.log(2 * np.pi) - np.sum(np.log(np.diag(C))) - z @ z / 2
    else:
        lp = (gammaln((df + n) / 2) - gammaln(df / 2) - n / 2 * np.log(df * np.pi)
              - np.sum(np.log(np.diag(C))) - (df + n) / 2 * np.log1p(z @ z / df))
    return lp


if __name__ == '__main__':
    main()
