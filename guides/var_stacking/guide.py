"""guide - the code of guides/var_stacking/README.md: a VAR(2) of three variables on generated
data, written as Y = Z @ A + U and as y = X @ beta + eps; their likelihoods, least squares by
equation against generalized least squares on the system, and the Kronecker form of
X' (I_T kron inv(Sig)) X; then one equation with fewer regressors. The sections called "the code
to adapt" contain everything a sampler uses. The sections called "the checks" form dense
matrices, which is practical only for a small example.

Saves fig_forms.png next to this file.

Usage, from the root of the repository:  python guides/var_stacking/guide.py
"""

import os

# one BLAS thread, set before NumPy loads (see Performance in the README of the repository)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import bvar_toolkit as bvt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def rel(a, b):
    """the largest difference between a and b, relative to the largest entry of b"""
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ---- 1. the data: a VAR(2) of three variables ----
n, p, T = 3, 2, 200
a = np.array([.5, .2, .1])                                  # intercepts
A1 = np.array([[.5, .1, 0], [.2, .4, .1], [0, .2, .5]])     # lag 1
A2 = np.array([[.2, 0, 0], [0, .1, 0], [.1, 0, .2]])        # lag 2
Sig = np.array([[1, .3, .2], [.3, .8, .1], [.2, .1, .6]])
A = np.vstack([a, A1.T, A2.T])     # k x n, A = (a, A_1, A_2)'
C = np.linalg.cholesky(Sig)
burn = 100
Yfull = np.zeros((burn + p + T, n))
for t in range(p, burn + p + T):
    Yfull[t] = a + A1 @ Yfull[t - 1] + A2 @ Yfull[t - 2] + C @ rng.standard_normal(n)
Yfull = Yfull[burn:]               # y_{-1}, y_0, y_1, ..., y_T

# ---- 2. the two forms: the code to adapt ----
Y, Z = bvt.util.build_lags(Yfull, p)   # T x n and T x k, k = 1 + n*p
k = Z.shape[1]
Ahat = np.linalg.lstsq(Z, Y, rcond=None)[0]   # least squares, one column per equation
U = Y - Z @ Ahat
Sighat = U.T @ U / T

y = Y.ravel()                      # y = vec(Y'): y_1, ..., y_T stacked
X = bvt.util.surform2(Z, n)        # rows t*n to t*n + n - 1 hold kron(I_n, x_t')
beta = bvt.util.vec(A)             # beta = vec(A)


# ---- 2. the checks ----
def loglik_mat(A, Sig):
    """the log likelihood in the matrix form"""
    E = Y - Z @ A
    return (-T * n / 2 * np.log(2 * np.pi) - T / 2 * bvt.util.ldet(Sig)
            - .5 * np.trace(np.linalg.solve(Sig, E.T @ E)))


def loglik_vec(b, Sig):
    """the log likelihood in the stacked form"""
    e = y - X @ b
    return (-T * n / 2 * np.log(2 * np.pi) - T / 2 * bvt.util.ldet(Sig)
            - .5 * e @ (sp.kron(sp.eye_array(T), np.linalg.inv(Sig)) @ e))


Zbyhand = np.column_stack([np.ones(T), Yfull[p - 1:-1], Yfull[p - 2:-2]])
lam = np.abs(np.linalg.eigvals(np.block([[A1, A2], [np.eye(n), np.zeros((n, n))]])))
print(f'\na VAR({p}) of {n} variables, T = {T}: k = {k} regressors per equation, '
      f'{n * k} coefficients')
print(f'largest modulus of the companion eigenvalues: {lam.max():.3f}')
dYZ = [np.max(np.abs(Y - Yfull[p:])), np.max(np.abs(Z - Zbyhand))]
print(f'Y and Z against direct indexing: {dYZ[0]:g}, {dYZ[1]:g} (largest difference)')
print(f'X: {X.shape[0]} x {X.shape[1]} with {X.nnz} nonzero entries, {X.nnz // T} per period')
gapX = rel(X @ beta, (Z @ A).ravel())
print(f"X @ vec(A) against vec((Z @ A)'): {gapX:.1e} (relative)")
ll = np.array([[loglik_mat(A, Sig), loglik_vec(beta, Sig)],            # matrix form, stacked
               [loglik_mat(Ahat, Sighat), loglik_vec(bvt.util.vec(Ahat), Sighat)]])
print(f'log likelihood at the true parameters:      {ll[0, 0]:.6f} (matrix form), '
      f'{ll[0, 1]:.6f} (stacked form)')
print(f'log likelihood at the least squares values: {ll[1, 0]:.6f} (matrix form), '
      f'{ll[1, 1]:.6f} (stacked form)')
assert all(d == 0 for d in dYZ) and gapX < 1e-12 and np.max(np.abs(ll[:, 0] - ll[:, 1])) < 1e-8, \
    'the matrix form and the stacked form disagree'
print(f'least squares estimates: largest error {np.max(np.abs(Ahat - A)):.3f}, '
      f'mean absolute error {np.mean(np.abs(Ahat - A)):.3f}')
print('the intercepts and lag-1 coefficients of equation 1 (true, then estimated):')
print(np.array2string(np.vstack([A[:4, 0], Ahat[:4, 0]]), precision=4, floatmode='fixed'))

# ---- 3. generalized least squares on the system: the code to adapt ----
W = sp.kron(sp.eye_array(T), np.linalg.inv(Sig), format='csc')   # the precision of eps
XWX = (X.T @ W @ X).toarray()      # nk x nk
XWy = X.T @ (W @ y)
bgls = np.linalg.solve(XWX, XWy)

# ---- 3. the checks ----
print('\ngeneralized least squares on the stacked form')
gapK = [rel(XWX, np.kron(np.linalg.inv(Sig), Z.T @ Z)),
        rel(XWy, bvt.util.vec(Z.T @ Y @ np.linalg.inv(Sig)))]
print(f"X' W X against kron(inv(Sig), Z' Z): {gapK[0]:.1e}; "
      f"X' W y against vec(Z' Y inv(Sig)): {gapK[1]:.1e}")
SigR = np.cov(rng.standard_normal((10, n)), rowvar=False)
gapG = []
for S, name in [(Sig, 'the true Sig'), (np.eye(n), 'Sig = I'), (SigR, 'a random Sig')]:
    WS = sp.kron(sp.eye_array(T), np.linalg.inv(S), format='csc')
    gapG.append(rel(np.linalg.solve((X.T @ WS @ X).toarray(), X.T @ (WS @ y)),
                    bvt.util.vec(Ahat)))
    print(f'GLS with {name} against least squares by equation: {gapG[-1]:.1e} (relative)')
Xd, Wd = X.toarray(), W.toarray()
gapD = rel(XWX, Xd.T @ Wd @ Xd)
print(f'the shortcut against dense algebra: {gapD:.1e}')
assert all(g < 1e-12 for g in gapK) and all(g < 1e-10 for g in gapG) and gapD < 1e-12, \
    'the Kronecker shortcut fails'

# ---- 4. one equation with fewer regressors: the adaptation ----
keep = np.setdiff1d(np.arange(n * k), [2, 3, 5, 6])   # equation 1 drops the lags of variables 2, 3
Xr = X[:, keep]
br_gls = np.linalg.solve((Xr.T @ W @ Xr).toarray(), Xr.T @ (W @ y))
br_ols = np.concatenate([np.linalg.lstsq(Z[:, [0, 1, 4]], Y[:, 0], rcond=None)[0],
                         bvt.util.vec(Ahat[:, 1:])])  # equation by equation
Wdiag = sp.kron(sp.eye_array(T), sp.diags_array(1 / np.diag(Sig)), format='csc')
br_gls_diag = np.linalg.solve((Xr.T @ Wdiag @ Xr).toarray(), Xr.T @ (Wdiag @ y))
print(f'\nequation 1 without the lags of variables 2 and 3: {keep.size} coefficients')
gapR = [np.max(np.abs(br_gls[:3] - br_ols[:3])), np.max(np.abs(br_gls[3:] - br_ols[3:])),
        np.max(np.abs(br_gls_diag - br_ols))]
print(f'GLS against least squares by equation, largest difference: {gapR[0]:.1e} in equation 1, '
      f'{gapR[1]:.3f} in equations 2 and 3')
print(f'the same with a diagonal Sig: {gapR[2]:.1e}')
assert gapR[0] < 1e-10 and gapR[2] < 1e-10, 'GLS differs from least squares where the two agree'

# ---- the figure ----
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.56))
for ax, M, title in [(axes[0], Z[:4], 'Z, t = 1, ..., 4'),
                     (axes[1], X[:4 * n].toarray(), 'X, t = 1, ..., 4')]:
    ax.spy(M, marker='.', markersize=4, color='k', aspect='auto')
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_ticks_position('bottom')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_forms.png'), dpi=150)
plt.close(fig)
