import numpy as np


def vec(Y):
    """bvt.util.vec - column-stacking operator: vec(Y) = Y.reshape(-1, order='F')."""
    return np.asarray(Y).reshape(-1, order='F')


def vech(Y):
    """bvt.util.vech - half-vectorization: the lower triangle of Y, stacked by column.

    NOTE the body keeps only the nonzero entries of the column-stacked tril(Y), so an exact zero
    in the lower triangle is DROPPED instead of stacked and the result is shorter than
    n(n+1)/2. Every caller in the toolkit passes a matrix whose lower triangle has no exact
    zeros.
    """
    v = np.tril(np.asarray(Y)).reshape(-1, order='F')
    return v[v != 0]
