"""The driver of the volatility-specification tutorial, tutorials/sv_specification/ml_varsv.py,
against run_ml of the Chan (2023) replication package in bvar-toolkit: the seven
specifications on the tutorial's data with short chains, draw for draw on the replay stream,
on which both languages take the mode of a generalized inverse Gaussian distribution in place
of a draw."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

import bvar_toolkit as bvt
from conftest import close, fixtures

TDIR = pathlib.Path(__file__).parents[1] / 'tutorials' / 'sv_specification'
sys.path.insert(0, str(TDIR))
import ml_varsv  # noqa: E402

FX = fixtures('drivers')
DATA = pd.read_csv(TDIR / 'macro5_Q.csv', float_precision='round_trip')[
    ['UNRATE', 'PCECTPI', 'FEDFUNDS', 'NFCI', 'GDPC1']].to_numpy(dtype=float)


@pytest.mark.parametrize('case', FX['run_ml'], ids=lambda c: f"{c['model']}-{c['seed']}")
def test_run_ml(case):
    r = bvt.Replay(case['seed'])
    out = ml_varsv.run_ml(case['model'], DATA, is_kappasym=bool(case['sym']),
                          nsim=int(case['nsim']), burnin=int(case['burnin']), rng=r,
                          r=int(case['r']), M=int(case['M']))
    assert r.count == case['count'], f'{r.count} uniforms used, MATLAB used {case["count"]}'
    # the marginal likelihood of VAR-SVO goes through a beta fit that stops within 1e-6 of its
    # optimum (see test_mlvarsv.py); the chains differ from MATLAB's by rounding that grows
    # with the sweeps
    close(out.lml, case['lml'], 1e-7)
    if 'lmlstd' in case:
        close(out.lmlstd, case['lmlstd'], 1e-4)
    for key in [k for k in case if k.startswith('mean_')]:
        close(np.squeeze(np.mean(out[key[5:]], axis=0)), case[key], 1e-5)
