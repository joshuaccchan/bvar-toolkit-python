"""Draws far in the tails: bvt.util.tnormrnd where both normal cdf values round to 1, and the KSC
samplers when every mixture density of some periods underflows."""

import numpy as np
from scipy.stats import norm

import bvar_toolkit as bvt


def test_tnormrnd_tails():
    """on (9, 10) and (-10, -9) the draws are finite, lie in the bounds and have the mean of the
    truncated normal"""
    N = 2000
    m = (norm.pdf(9) - norm.pdf(10)) / (norm.sf(9) - norm.sf(10))   # the mean of N(0,1) on (9, 10)
    t = bvt.util.tnormrnd(0, 1, 9, 10, N, rng=5)
    assert np.all(np.isfinite(t) & (t >= 9) & (t <= 10))
    assert abs(t.mean() - m) < 4 * t.std() / np.sqrt(N)
    t = bvt.util.tnormrnd(0, 1, -10, -9, N, rng=5)
    assert np.all(np.isfinite(t) & (t >= -10) & (t <= -9))
    assert abs(t.mean() + m) < 4 * t.std() / np.sqrt(N)


def test_ksc_underflow():
    """the paths stay finite, and the periods whose densities all underflow take the widest
    component, whose density is largest there by thousands of log points"""
    r = np.random.default_rng(3)
    T = 60
    ystar = np.log(r.standard_normal(T)**2 + 1e-4)
    h = np.zeros(T)
    h[9:12] = 2000.0                     # every density underflows in periods 10 to 12
    h1, S1 = bvt.sv.ksc_ar1_mean(ystar, h, 0, .9, .1, rng=1)
    h2, S2 = bvt.sv.ksc_rw_diffuse(ystar, h, .1, 10, rng=1)
    h3 = bvt.sv.ksc_rw_h0(ystar, h, .1, 0, rng=1)
    assert np.all(np.isfinite(np.r_[h1, h2, h3]))
    assert np.all(np.isin(np.r_[S1, S2], np.arange(7)))
    assert np.all(S1[9:12] == 0) and np.all(S2[9:12] == 0)
