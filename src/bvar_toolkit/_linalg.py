"""Banded Cholesky factors for the precision samplers, and the MA(1) solve of the MA models."""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky_banded
from scipy.linalg.lapack import dtbtrs, dtrtrs
from scipy.signal import lfilter


def ma_solve(psi, U):
    """The solution X of Hpsi X = U for Hpsi = I + psi L, L the lag matrix, by forward
    substitution: row t of X is U[t] - psi times row t-1 of X, as MATLAB's sparse triangular
    solve computes it. U is a vector or has one column per series."""
    return lfilter([1.0], [1.0, psi], np.asarray(U, dtype=float), axis=0)


class BandChol:
    """The lower Cholesky factor C of a symmetric positive definite K = C @ C.T, held in LAPACK
    band storage, as MATLAB's chol(K, 'lower') for a sparse banded K, with no reordering. Only
    the lower triangle of K is read; K may be sparse or dense.

    A right-hand side with one or a few columns is solved by LAPACK's banded triangular solver,
    one column at a time; one with many columns by rows of blocks of the factor, with dense
    matrix products, which is faster for the same result up to rounding."""

    MANY = 8           # columns from which the blocked solver is used

    def __init__(self, K):
        if sp.issparse(K):
            K = sp.coo_array(K)
            K.sum_duplicates()
            low = K.row >= K.col
            i, j, v = K.row[low], K.col[low], K.data[low]
        else:
            K = np.asarray(K, dtype=float)
            i, j = np.nonzero(np.tril(K))
            v = K[i, j]
        N = K.shape[0]
        kd = int(np.max(i - j)) if i.size else 0
        ab = np.zeros((kd + 1, N))
        ab[i - j, j] = v
        self.ab = cholesky_banded(ab, lower=True)
        self._blocks = None

    @classmethod
    def from_band(cls, ab):
        """The factor of the matrix whose lower band storage is ab: ab[i-j, j] = K[i, j]."""
        self = cls.__new__(cls)
        self.ab = cholesky_banded(np.asarray(ab, dtype=float), lower=True)
        self._blocks = None
        return self

    def _solve(self, b, trans):
        b = np.asarray(b, dtype=float)
        if b.ndim == 2 and b.shape[1] >= self.MANY:
            return self._solve_blocked(b, trans)
        x, info = dtbtrs(self.ab, b.reshape(b.shape[0], -1), uplo='L', trans=trans)
        if info != 0:
            raise np.linalg.LinAlgError(f'the Cholesky factor is singular (dtbtrs info {info})')
        return x.reshape(b.shape)

    def _dense(self, r0, r1, c0, c1):
        """C[r0:r1, c0:c1] as a dense matrix."""
        kd = self.ab.shape[0] - 1
        r = np.arange(r0, r1)[:, None]
        c = np.arange(c0, c1)[None, :]
        off = r - c
        inside = (off >= 0) & (off <= kd)
        out = np.zeros((r1 - r0, c1 - c0))
        out[inside] = self.ab[off[inside], np.broadcast_to(c, off.shape)[inside]]
        return out

    def _solve_blocked(self, B, trans):
        """C \\ B or C.T \\ B by blocks of s >= kd rows, so that each block of C couples only
        with the block before it."""
        N = self.ab.shape[1]
        if self._blocks is None:
            s = max(self.ab.shape[0] - 1, 64)
            starts = list(range(0, N, s))
            D = [self._dense(i0, min(i0 + s, N), i0, min(i0 + s, N)) for i0 in starts]
            E = [None] + [self._dense(i0, min(i0 + s, N), i0 - s, i0) for i0 in starts[1:]]
            self._blocks = (s, starts, D, E)
        s, starts, D, E = self._blocks
        X = np.empty_like(B)
        if trans == 'N':
            for k, i0 in enumerate(starts):
                i1 = min(i0 + s, N)
                rhs = B[i0:i1] if k == 0 else B[i0:i1] - E[k] @ X[i0 - s:i0]
                X[i0:i1] = dtrtrs(D[k], rhs, lower=1)[0]
        else:
            for k in range(len(starts) - 1, -1, -1):
                i0 = starts[k]
                i1 = min(i0 + s, N)
                if k == len(starts) - 1:
                    rhs = B[i0:i1]
                else:
                    rhs = B[i0:i1] - E[k + 1].T @ X[i1:min(i1 + s, N)]
                X[i0:i1] = dtrtrs(D[k], rhs, lower=1, trans=1)[0]
        return X

    def solve(self, b):
        """C \\ b, for a vector or a matrix b."""
        return self._solve(b, 'N')

    def solve_t(self, b):
        """C.T \\ b, for a vector or a matrix b."""
        return self._solve(b, 'T')

    def logdet(self):
        """log det K."""
        return 2.0 * np.sum(np.log(self.ab[0]))
