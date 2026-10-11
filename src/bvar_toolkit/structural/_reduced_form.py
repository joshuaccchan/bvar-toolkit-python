import numpy as np


def reduced_form(store_alp, store_beta, store_Sig):
    """bvt.structural.reduced_form - map stored structural-form draws of the
    asymmetric-conjugate VAR to their reduced form.

      store_Btilde, store_Sigtilde = bvt.structural.reduced_form(store_alp, store_beta,
                                                                 store_Sig)

      store_alp      : nsim x n(n-1)/2 free elements of the unit-lower-triangular A, stacked
                       equation by equation
      store_beta     : nsim x (n^2 p + n) structural coefficients, equation by equation
      store_Sig      : nsim x n structural innovation variances (diagonal)
      store_Btilde   : nsim x (n^2 p + n), the reduced-form coefficients stacked the same way
      store_Sigtilde : nsim x n x n reduced-form covariances

    The structural form is A y_t = B x_t + eps_t with A unit lower triangular and
    Var(eps_t) = diag(sig). The reduced form follows by inversion:
    Sigtilde = A^{-1} diag(sig) A^{-T} and Btilde = (A^{-1} B')'. Both are formed with
    np.linalg.solve, for all draws at once.

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169.
    """
    store_alp = np.asarray(store_alp, dtype=float)
    store_beta = np.asarray(store_beta, dtype=float)
    store_Sig = np.asarray(store_Sig, dtype=float)
    nsim, n = store_Sig.shape
    kb = store_beta.shape[1] // n
    rows, cols = np.tril_indices(n, -1)             # row by row, as the draws are stacked
    A = np.tile(np.eye(n), (nsim, 1, 1))
    A[:, rows, cols] = store_alp.reshape(nsim, -1)
    X = np.linalg.solve(A, store_Sig[:, :, None] * np.eye(n))            # A \ diag(sig)
    store_Sigtilde = np.linalg.solve(A, X.transpose(0, 2, 1)).transpose(0, 2, 1)   # ... / A'
    store_Btilde = np.linalg.solve(A, store_beta.reshape(nsim, n, kb)).reshape(nsim, n * kb)
    return store_Btilde, store_Sigtilde
