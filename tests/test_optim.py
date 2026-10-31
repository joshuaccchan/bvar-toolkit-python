"""fminbnd, fminsearch and fzero against MATLAB's, on the objectives of export_fixtures.m
written with the same order of operations, the gamma and beta fits against MATLAB's gamfit
and betafit, the sum and gammaln they use and linspace against MATLAB's, digit for digit, and
the stand-in for fminunc against fminunc."""

import numpy as np
import pytest
from scipy.optimize import brentq

from bvar_toolkit._optim import fminbnd, fminsearch, fminunc, fzero
from bvar_toolkit._stats import betafit, gamfit, matlab_gammaln, matlab_linspace, matlab_sum
from conftest import cases, close, fixtures, inputs, outputs

FX = fixtures('optim')

OBJECTIVES = {
    'f1': lambda x: (x - .3) ** 2 * (x + .5) + .01 / x,
    'f2': lambda x: x * x * x * x - 3 * x * x * x + 2 * x + 1,
    'rosen': lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
    'quad3': lambda x: ((x[0] - 1) ** 2 + 10 * (x[1] - 2) ** 2 + 100 * (x[2] + .5) ** 2
                        + x[0] * x[1]),
    'g1': lambda x: x * x * x - 2 * x - 5,
    'g2': lambda x: np.cos(x) - x,
    'g3': lambda x: x * x - 1e-6,
    'u1': lambda x: .5 * (x - .25) ** 2 / .01 + .5 * np.log(1 + x ** 2),
    'u2': lambda x: 3 * np.log(1 + (x - .4) ** 2 / .2) + .5 * x ** 2,
}


@pytest.mark.parametrize('case', cases(FX, 'fminbnd'))
def test_fminbnd(case):
    a, b = inputs(case)
    x, fval = fminbnd(OBJECTIVES[case['name']], a, b)
    xm, fm = outputs(case)
    close(x, xm, 1e-14)
    close(fval, fm, 1e-14)


@pytest.mark.parametrize('case', cases(FX, 'fminsearch'))
def test_fminsearch(case):
    x0 = np.asarray(inputs(case)[0], dtype=float)
    x, fval = fminsearch(OBJECTIVES[case['name']], x0)
    xm, fm = outputs(case)
    close(x, xm, 1e-14)
    close(fval, fm, 1e-14)


@pytest.mark.parametrize('case', cases(FX, 'fzero'))
def test_fzero(case):
    bracket = np.asarray(inputs(case)[0], dtype=float)
    x, fval = fzero(OBJECTIVES[case['name']], bracket)
    xm, fm = outputs(case)
    close(x, xm, 1e-14)
    close(fval, fm, 1e-14)


@pytest.mark.parametrize('name', ['gamfit', 'betafit'])
def test_fits(name):
    fit = {'gamfit': gamfit, 'betafit': betafit}[name]
    for case in cases(FX, name):
        close(fit(inputs(case)[0]), outputs(case)[0], 1e-12)


def test_matlab_sum():
    for case in cases(FX, 'sum'):
        assert matlab_sum(inputs(case)[0]) == outputs(case)[0]


def test_matlab_gammaln():
    # equal where gammaln takes no logarithm; elsewhere NumPy's log and MATLAB's can differ in
    # the last digit, which moves gammaln by up to two units in the last place
    x, = inputs(FX['gammaln'])
    g, gm = matlab_gammaln(x), outputs(FX['gammaln'])[0]
    nolog = ((x >= .6796875) & (x <= 12)) | ((x == np.floor(x)) & (x >= 1) & (x <= 100))
    np.testing.assert_array_equal(g[nolog], gm[nolog])
    np.testing.assert_array_equal(np.isinf(g), np.isinf(gm))
    fin = np.isfinite(gm)
    assert np.all(np.abs(g[fin] - gm[fin]) <= 2 * np.spacing(np.abs(gm[fin])))


def test_matlab_linspace():
    for case in cases(FX, 'linspace'):
        d1, d2, n = inputs(case)
        np.testing.assert_array_equal(matlab_linspace(d1, d2, n), np.ravel(outputs(case)[0]))


DERIVATIVES = {
    'u1': lambda x: (x - .25) / .01 + x / (1 + x ** 2),
    'u2': lambda x: 3 * (2 * (x - .4) / .2) / (1 + (x - .4) ** 2 / .2) + x,
}


@pytest.mark.parametrize('case', cases(FX, 'fminunc'))
def test_fminunc(case):
    """The Nelder-Mead search ends within 2e-8 of the minimizer that fminunc finds, and nearer
    the exact minimizer; the second derivative there agrees with fminunc's to 1e-8."""
    x0, = inputs(case)
    x, fval, hess = fminunc(OBJECTIVES[case['name']], x0)
    xm, fm, hm = outputs(case)
    exact = brentq(DERIVATIVES[case['name']], 0, .6, xtol=1e-16)
    assert abs(x - xm) <= 2e-8
    assert abs(x - exact) <= abs(xm - exact)
    close(fval, fm, 1e-12)
    close(hess, hm, 1e-8)
