"""fminbnd and fminsearch against MATLAB's, on the objectives of export_fixtures.m written
with the same order of operations."""

import numpy as np
import pytest

from bvar_toolkit._optim import fminbnd, fminsearch
from conftest import cases, close, fixtures, inputs, outputs

FX = fixtures('optim')

OBJECTIVES = {
    'f1': lambda x: (x - .3) ** 2 * (x + .5) + .01 / x,
    'f2': lambda x: x * x * x * x - 3 * x * x * x + 2 * x + 1,
    'rosen': lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
    'quad3': lambda x: (x[0] - 1) ** 2 + 10 * (x[1] - 2) ** 2 + 100 * (x[2] + .5) ** 2 + x[0] * x[1],
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
