import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

FX = fixtures('diag')


def lag_int(args, kwargs):
    if len(args) > 1 and not isinstance(args[1], str):
        args[1] = int(args[1])
    return args, kwargs


@pytest.mark.parametrize('name,case', [(n, c) for n in ['specvar0', 'inefficiency_factor', 'mcse']
                                       for c in cases(FX, n)])
def test_against_matlab(name, case):
    check(getattr(bvt.diag, name), case, lag_int)


@pytest.mark.parametrize('case', cases(FX, 'geweke'))
def test_geweke(case):
    def adapt(args, kwargs):
        if len(args) > 3:
            args[3] = int(args[3])
        return args, kwargs
    Z, pval, info = run(bvt.diag.geweke, case, adapt)
    Zm, pm, im = outputs(case)
    close(Z, Zm)
    close(pval, pm)
    close(info['A'], np.asarray(im['A']) - 1)       # row indices count from 0 here
    close(info['B'], np.asarray(im['B']) - 1)
    assert (info['LA'], info['LB'], info['a'], info['b']) == (im['LA'], im['LB'], im['a'], im['b'])


def test_specvar0_rejects_bad_lag():
    with pytest.raises(ValueError):
        bvt.diag.specvar0(np.ones(10), 10)
