"""Real-time forecast evaluation: the estimation panel and realized values of a data vintage,
one iterated forecast and its scores per posterior draw under each error specification, and
the tables of forecast errors and log predictive likelihoods, compared with MATLAB draw for
draw."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, inputs, outputs

FC = fixtures('forecast')


@pytest.mark.parametrize('case', cases(FC, 'realtime_loaddata'))
def test_realtime_loaddata(case):
    def adapt(args, kwargs):
        args[2], args[3] = int(args[2]), int(args[3])
        return args, kwargs
    check(bvt.forecast.realtime_loaddata, case, adapt)


@pytest.mark.parametrize('case', cases(FC, 'iterate'),
                         ids=lambda c: f"{c['in1']}-{int(c['seed'])}")
def test_iterate(case):
    def adapt(args, kwargs):
        cfg = dict(args[2])
        cfg['p'], cfg['t'], cfg['T'] = int(cfg['p']), int(cfg['t']), int(cfg['T'])
        args[2] = cfg
        return args, kwargs
    check(bvt.forecast.iterate, case, adapt)


@pytest.mark.parametrize('case', cases(FC, 'tables'), ids=lambda c: c['in1'])
def test_tables(case):
    args = inputs(case)
    if args[0] == 'springer' and np.size(args[3]):
        args[3] = np.asarray(args[3], dtype=int) - 1        # positions count from 0
    out = bvt.forecast.tables(*args)
    expected, = outputs(case)
    if isinstance(expected, dict):
        assert set(out) == set(expected)
        for field, value in expected.items():
            close(out[field], value)
    else:
        close(out, expected)
