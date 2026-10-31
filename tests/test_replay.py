"""The replay stream: each draw of bvt.Replay equals the draw of its MATLAB twin in
tools/matlab/replay on the same seed, and uses as many uniforms."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import close, fixtures, inputs, outputs

FX = fixtures('replay')
CASES = [(name, c) for name in ('randn', 'rand', 'gamrnd', 'betarnd', 'unidrnd', 'randi',
                                'randperm', 'iwishrnd')
         for c in (FX[name] if isinstance(FX[name], list) else [FX[name]])]


@pytest.mark.parametrize('name, case', CASES, ids=[f'{n}-{c["seed"]}' for n, c in CASES])
def test_replay(name, case):
    r = bvt.Replay(case['seed'])
    args = [int(a) if np.ndim(a) == 0 and float(a).is_integer() and name != 'iwishrnd' else a
            for a in inputs(case)]
    out = getattr(r, name)(*args)
    expected, = outputs(case)
    if name == 'randperm':
        out = np.asarray(out) + 1                  # MATLAB counts from 1
    close(out, expected)
    assert r.count == case['count']
