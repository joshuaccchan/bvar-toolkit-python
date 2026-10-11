"""Shared helpers: the test cases exported from the MATLAB toolkit by
tools/matlab/export_fixtures.m, and the comparison of a result with its MATLAB twin."""

import pathlib

import numpy as np
import scipy.io
import scipy.sparse as sp

import bvar_toolkit as bvt

FIXTURES = pathlib.Path(__file__).parent / 'fixtures'
TOL = 1e-10


def fixtures(namespace):
    """The cases of one namespace, keyed by function name."""
    return scipy.io.loadmat(FIXTURES / f'{namespace}.mat', simplify_cells=True)


def cases(fx, name):
    """The cases of one function, as a list for pytest.mark.parametrize."""
    c = fx[name]
    return [c] if isinstance(c, dict) else list(c)


def inputs(case):
    return [case[f'in{i}'] for i in range(1, int(case['nin']) + 1)]


def outputs(case):
    return [case[f'out{i}'] for i in range(1, int(case['nout']) + 1)]


def dense(x):
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=float)


def close(py, ml, tol=TOL):
    """Assert that a result equals its MATLAB twin entry by entry: within tol of the entry
    itself plus 1e-14 of the largest entry, which leaves room for rounding in entries near
    zero. NaN and infinite entries must match. Shapes may differ only by singleton
    dimensions, which loadmat squeezes."""
    a, b = dense(py), dense(ml)
    assert [s for s in a.shape if s != 1] == [s for s in b.shape if s != 1], (a.shape, b.shape)
    a, b = a.ravel(), b.ravel()
    fin = np.isfinite(b)
    assert np.array_equal(np.isnan(a), np.isnan(b)), 'NaN entries differ'
    assert np.array_equal(a[np.isinf(b)], b[np.isinf(b)]), 'infinite entries differ'
    if fin.any():
        scale = max(1.0, np.max(np.abs(b[fin])))
        bound = tol * np.abs(b[fin]) + 1e-14 * scale
        worst = np.max(np.abs(a[fin] - b[fin]) / bound)
        assert worst <= 1, f'a difference is {worst:.3g} times the bound of tol = {tol:g}'


def run(fn, case, adapt=None):
    """Call fn on the inputs of a case, with the replay stream when the case has a seed,
    and return the outputs as a list; adapt maps (args, kwargs) to the Python call."""
    args = inputs(case)
    kwargs = dict(case['opts']) if isinstance(case.get('opts'), dict) else {}
    if adapt is not None:
        args, kwargs = adapt(args, kwargs)
    r = None
    if 'seed' in case:
        r = bvt.Replay(case['seed'])
        kwargs['rng'] = r
    out = fn(*args, **kwargs)
    if r is not None:
        assert r.count == case['count'], f'{r.count} uniforms used, MATLAB used {case["count"]}'
    return list(out) if isinstance(out, tuple) else [out]


def check(fn, case, adapt=None, tol=TOL):
    """Run a case and compare every output with MATLAB's."""
    out = run(fn, case, adapt)
    expected = outputs(case)
    assert len(out) == len(expected)
    for o, e in zip(out, expected):
        close(o, e, tol)
