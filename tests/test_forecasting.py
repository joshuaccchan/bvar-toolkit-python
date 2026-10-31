"""The VAR with common stochastic volatility and its forecasts: the natural conjugate prior,
the one-block update of the volatility path, short chains of the complete sampler, the
simulated predictive distribution of one draw under each covariance specification, and the
quantiles of a mixture of normals."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

PR = fixtures('priors')
SV = fixtures('sv')
MO = fixtures('models')
FC = fixtures('forecast')


@pytest.mark.parametrize('case', cases(PR, 'niw'))
def test_niw(case):
    def adapt(args, kwargs):
        args[0] = int(args[0])
        return args, kwargs
    check(bvt.priors.niw, case, adapt)


def test_niw_needs_four_lags_but_in_mlvarsv_ncp():
    """The AR(4) fits prepend the last p rows of Y0, which is conformable only for p = 4,
    except under 'mlvarsv_ncp', which prepends the last four rows whatever p is."""
    rng = np.random.default_rng(0)
    Y0, Y = rng.standard_normal((8, 3)), rng.standard_normal((40, 3))
    with pytest.raises(ValueError):
        bvt.priors.niw(2, [.04, 100], Y0, Y, 'largebvar_nc')
    assert bvt.priors.niw(2, [.04, 100], Y0, Y, 'mlvarsv_ncp')[1].size == 7


@pytest.mark.parametrize('case', cases(SV, 'csv_armh'))
def test_csv_armh(case):
    check(bvt.sv.csv_armh, case)


@pytest.mark.parametrize('case', cases(MO, 'var_csv'))
def test_var_csv(case):
    def adapt(args, kwargs):
        args[2] = int(args[2])
        return args, kwargs
    res, = run(bvt.models.var_csv, case, adapt)
    ml, = outputs(case)
    for f in ('A_mean', 'Sig_mean', 'h_mean', 'phi_mean', 'sigh2_mean', 'accept_rate'):
        close(res[f], ml[f])
    if 'draws' in ml:
        for f in ml['draws']:
            close(res.draws[f], ml['draws'][f])


def test_var_csv_counts_sweeps_after_20(monkeypatch):
    """The first 20 sweeps take the proposal of h regardless, and only the later ones count
    toward accept_rate."""
    from bvar_toolkit.models import _var_csv as mod
    real = mod.csv_armh
    forced = []

    def stub(*args, **kwargs):
        forced.append(args[5])
        h, _ = real(*args, **kwargs)
        return h, len(forced) % 2                # odd sweeps accept, even ones reject
    monkeypatch.setattr(mod, 'csv_armh', stub)
    case = cases(MO, 'var_csv')[0]
    res = bvt.models.var_csv(case['in1'], case['in2'], 2, nsim=7, burnin=18, rng=1)
    assert forced == [True] * 20 + [False] * 5
    assert res.accept_rate == 3 / 5                # sweeps 21, 23 and 25


@pytest.mark.parametrize('case', cases(FC, 'simulate'))
def test_simulate(case):
    check(bvt.forecast.simulate, case)


@pytest.mark.parametrize('case', cases(FC, 'mixquantile'))
def test_mixquantile(case):
    check(bvt.forecast.mixquantile, case)
