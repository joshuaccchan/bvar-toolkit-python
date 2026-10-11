import numpy as np


def gam_mode(store_gam):
    """bvt.util.gam_mode - the modal configuration of a stored binary indicator matrix: the row
    value that occurs most often across draws. Used on the hybrid TVP-VAR's store_gam, whose
    nsim x 2n rows record which equations had time-varying VAR coefficients and which had
    time-varying impact elements, to report the single most frequent model.

      gam_mode = bvt.util.gam_mode(store_gam)

      store_gam : nsim x nmodel matrix of 0/1 draws
      gam_mode  : nmodel-vector, the most frequent configuration

    Each row is keyed by its binary expansion gam @ 2**arange(nmodel), counts are accumulated
    in a dictionary, and ties are broken in favor of the configuration seen first.

    See:
    Chan, J.C.C. (2023). Large Hybrid Time-Varying Parameter VARs, Journal of Business and
    Economic Statistics, 41(3): 890-905.
    """
    store_gam = np.asarray(store_gam, dtype=float)
    nsim, nmodel = store_gam.shape
    w = 2.0 ** np.arange(nmodel)
    first = {}                                      # model id -> first row seen
    counts = {}
    for ii in range(nsim):
        idx = store_gam[ii] @ w + 1
        if idx in counts:
            counts[idx] += 1
        else:
            counts[idx] = 1
            first[idx] = store_gam[ii]
    ids = list(counts)
    best = ids[int(np.argsort(-np.array([counts[i] for i in ids]), kind='stable')[0])]
    return first[best].copy()
