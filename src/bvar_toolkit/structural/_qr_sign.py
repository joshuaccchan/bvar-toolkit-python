import numpy as np


def qr_sign(A):
    """bvt.structural.qr_sign - QR decomposition with the diagonal of R normalized to be
    positive, which makes the decomposition unique and gives the orthogonal factor Q used to
    draw rotations in sign-restricted SVARs.

      Q, R = bvt.structural.qr_sign(A)

    For A = rng.randn(n, n), Q is drawn from the uniform (Haar) measure on the orthogonal
    group. Callers use it to rotate the Cholesky factor of the reduced-form covariance.

    See:
    Chan, J.C.C. (2022). Asymmetric Conjugate Priors for Large Bayesian VARs, Quantitative
    Economics, 13(3): 1145-1169.
    """
    Q, R = np.linalg.qr(np.asarray(A, dtype=float))
    d = np.sign(np.diag(R))
    return Q * d[None, :], d[:, None] * R
