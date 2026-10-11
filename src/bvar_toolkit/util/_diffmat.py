import numpy as np
import scipy.sparse as sp


def diffmat(T, a=1.0):
    """bvt.util.diffmat - the sparse first-order difference matrix of a state equation,
    H = I_T - a*L, where L is the first subdiagonal (the lag operator on a stacked path).

      H = bvt.util.diffmat(T)        # a = 1: the random-walk difference matrix
      H = bvt.util.diffmat(T, a)

      T : path length, a positive integer
      a : scalar autoregressive coefficient (default 1)
      H : T x T sparse lower bidiagonal matrix (csc_array)

    This matrix writes a state equation as one linear system in the whole path, the form the
    precision-based samplers use: for x_t = a*x_{t-1} + u_t with u ~ N(0, S), H @ x = u, so the
    path has precision H.T @ inv(S) @ H - banded, hence an O(T) Cholesky.

    Sign conventions in this toolkit, both of which appear in the papers:
      AR(1) / random-walk state:  H_rho  = I - rho*L   ->  diffmat(T, rho)
      MA(1) error transform:      H_psi  = I + psi*L   ->  diffmat(T, -psi)
    """
    if not np.isscalar(T) or T < 1 or T != int(T):
        raise ValueError('T must be a positive integer')
    if not np.isscalar(a):
        raise ValueError('a must be a scalar')
    T = int(T)
    if T == 1:
        return sp.eye_array(1, format='csc')
    L = sp.csc_array((np.ones(T - 1), (np.arange(1, T), np.arange(T - 1))), shape=(T, T))
    return (sp.eye_array(T, format='csc') - a * L).tocsc()
