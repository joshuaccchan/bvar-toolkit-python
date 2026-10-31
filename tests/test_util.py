import numpy as np
import pandas as pd
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, fixtures

FX = fixtures('util')


def ints(*pos):
    """An adapter turning the arguments at these positions into integers."""
    def adapt(args, kwargs):
        for i in pos:
            if i < len(args):
                args[i] = int(args[i])
        return args, kwargs
    return adapt


SIMPLE = {
    'build_lags': (bvt.util.build_lags, ints(1)),
    'diffmat': (bvt.util.diffmat, ints(0)),
    'dirirnd': (bvt.util.dirirnd, ints(1)),
    'dlog_gaps': (bvt.util.dlog_gaps, None),
    'gam_mode': (bvt.util.gam_mode, None),
    'igrnd': (bvt.util.igrnd, None),
    'kron_gram': (bvt.util.kron_gram, None),
    'ldet': (bvt.util.ldet, None),
    'mgammaln': (bvt.util.mgammaln, ints(0)),
    'select_obs': (bvt.util.select_obs, None),
    'simulate_var': (bvt.util.simulate_var, ints(2)),
    'surform': (bvt.util.surform, None),
    'surform2': (bvt.util.surform2, ints(1)),
    'tnormrnd': (bvt.util.tnormrnd, ints(4)),
    'anormrnd': (bvt.util.anormrnd, None),
    'vec': (bvt.util.vec, None),
    'vech': (bvt.util.vech, None),
}


@pytest.mark.parametrize('name,case', [(n, c) for n in SIMPLE for c in cases(FX, n)])
def test_against_matlab(name, case):
    fn, adapt = SIMPLE[name]
    check(fn, case, adapt)


@pytest.mark.parametrize('case', cases(FX, 'logsumexp'))
def test_logsumexp(case):
    def adapt(args, kwargs):
        if len(args) > 1:
            kwargs['axis'] = int(args.pop()) - 1
        return args, kwargs
    check(bvt.util.logsumexp, case, adapt)


@pytest.mark.parametrize('case', cases(FX, 'mm_constraint'))
def test_mm_constraint(case):
    def adapt(args, kwargs):
        args[1] = np.asarray(args[1], dtype=bool)
        return args, kwargs
    check(bvt.util.mm_constraint, case, adapt)


def test_logsumexp_all_minus_inf():
    assert bvt.util.logsumexp(np.array([-np.inf, -np.inf])) == -np.inf


def test_build_lags_errors():
    with pytest.raises(ValueError):
        bvt.util.build_lags(np.ones((5, 2)), 0)
    with pytest.raises(ValueError):
        bvt.util.build_lags(np.ones((3, 2)), 3)


def test_surform2_is_the_stacked_regression():
    rng = np.random.default_rng(0)
    T, n, k = 6, 3, 4
    Z = rng.standard_normal((T, k))
    A = rng.standard_normal((k, n))
    X = bvt.util.surform2(Z, n)
    assert np.allclose(X @ bvt.util.vec(A), (Z @ A).ravel())


def test_report(tmp_path):
    tbl = pd.DataFrame({'prior': ['a', 'b'], 'log_ML': [-1.5, -2.5]})
    files = bvt.util.report('r', {'models': tbl}, {'p': 4, 'kappa': np.array([.1, .2])}, tmp_path)
    assert [f.split('\\')[-1].split('/')[-1] for f in files] == ['r_models.csv', 'r.json']
    back = pd.read_csv(files[0])
    assert back.equals(tbl)
    import json
    meta = json.loads(open(files[1], encoding='utf-8').read())
    assert meta['kappa'] == [.1, .2] and meta['tables'] == ['r_models.csv']


def test_shaded_band():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    h = bvt.util.shaded_band(np.arange(5), np.zeros(5), np.ones(5), ax=ax)
    assert h is not None
    plt.close(fig)
