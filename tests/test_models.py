"""Complete samplers: short chains of the mixed-frequency VAR with common stochastic
volatility, compared with MATLAB draw for draw."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, close, fixtures, outputs, run

FX = fixtures('models')

# the draws of the missing values are kept in single precision, so a draw and the quantiles
# computed from the draws can differ from MATLAB in the last bit of a float32
SINGLE = 1e-6


@pytest.mark.parametrize('case', cases(FX, 'mfvar_csv'))
def test_mfvar_csv(case):
    def adapt(args, kwargs):
        args[1] = int(args[1])
        return args, kwargs
    res, = run(bvt.models.mfvar_csv, case, adapt)
    ml, = outputs(case)
    for f in ('Y_mean', 'A_mean', 'Sig_mean', 'h_mean', 'phi_mean', 'sigh2_mean', 'accept_rate',
              'sig2'):
        close(res[f], ml[f])
    close(res.Y_q, ml['Y_q'], SINGLE)
    close(res.prior.V, ml['prior']['V'])
    assert res.max_resid < 1e-10 and ml['max_resid'] < 1e-10
    if 'draws' in ml:
        for f in ('A', 'Sig', 'h', 'phi', 'sigh2'):
            close(res.draws[f], ml['draws'][f])
        assert res.draws.ym.dtype == np.float32
        close(res.draws.ym, ml['draws']['ym'], SINGLE)


def test_mfvar_csv_counts_blocks_after_20_sweeps(monkeypatch):
    """The first 20 sweeps take the proposal of every block of h regardless, and only the
    later ones count toward accept_rate."""
    from bvar_toolkit.models import _mfvar_csv as mod
    real = mod.csv_armh_block
    forced = []

    def stub(*args, **kwargs):
        forced.append(kwargs['forced_accept'])
        h, _, _ = real(*args, **kwargs)
        return h, len(forced) % 2, 1             # odd sweeps accept, even ones reject
    monkeypatch.setattr(mod, 'csv_armh_block', stub)
    case = cases(FX, 'mfvar_csv')[0]
    Y, p = case['in1'], int(case['in2'])
    opts = {k: case['opts'][k] for k in ('M', 'z', 'sig2', 'hblock')}
    res = bvt.models.mfvar_csv(Y, p, nsim=7, burnin=18, rng=1, **opts)
    assert forced == [True] * 20 + [False] * 5
    assert res.accept_rate == 3 / 5                # sweeps 21, 23 and 25


def test_mfvar_csv_needs_sig2_of_a_quarterly_series():
    Y = np.random.default_rng(0).standard_normal((40, 2))
    Y[:, 1] = np.nan
    with pytest.raises(ValueError, match='sig2 must be given'):
        bvt.models.mfvar_csv(Y, 2, nsim=1, burnin=0)
