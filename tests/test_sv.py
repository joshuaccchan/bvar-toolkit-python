"""Stochastic volatility: the block update of the common volatility path, the draw of the
state-equation parameters of zero-mean log-volatilities, on every branch of the truncated
normal candidate, the auxiliary mixture sampler for a random walk, and the Student-t degrees
of freedom."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, grid, outputs, run

FX = fixtures('sv')


@pytest.mark.parametrize('case', cases(FX, 'csv_armh_block'))
def test_csv_armh_block(case):
    def adapt(args, kwargs):
        names = {'ForcedAccept': 'forced_accept', 'MaxIterMode': 'max_iter_mode',
                 'MaxIterAR': 'max_iter_ar'}
        return args, {names.get(k, k): v for k, v in kwargs.items()}
    check(bvt.sv.csv_armh_block, case, adapt)


@pytest.mark.parametrize('case', cases(FX, 'sv0_params'))
def test_sv0_params(case):
    check(bvt.sv.sv0_params, case)


def test_sv0_params_branches():
    """The cases reach each way of drawing the truncated candidate: the inverse transform
    below the mean, above it and around it, and the tail sampler on each side."""
    from bvar_toolkit.sv import _sv0_params as mod
    seen = set()
    draw, tail = mod._tnorm_draw, mod._tail_draw

    def tnorm(mu, s, a, b, r):
        alpha, beta = (a - mu) / s, (b - mu) / s
        seen.add('below' if beta < 0 else 'above' if alpha > 0 else 'around')
        return draw(mu, s, a, b, r)

    def tail_draw(c, d, r):
        seen.add('tail')
        return tail(c, d, r)

    mod._tnorm_draw, mod._tail_draw = tnorm, tail_draw
    try:
        for case in cases(FX, 'sv0_params'):
            if isinstance(case['opts'], dict) and case['opts'].get('proposal') == 'truncated':
                check(bvt.sv.sv0_params, case)
    finally:
        mod._tnorm_draw, mod._tail_draw = draw, tail
    assert seen == {'below', 'above', 'around', 'tail'}


def test_csv_armh_block_errors():
    with pytest.raises(ValueError, match='block'):
        bvt.sv.csv_armh_block(np.ones(10), .9, .1, np.zeros(10), 2, block=0)
    with pytest.raises(ValueError, match='c_reject'):
        bvt.sv.csv_armh_block(np.ones(10), .9, .1, np.zeros(10), 2, c_reject=-1)


@pytest.mark.parametrize('case', cases(FX, 'ksc_rw_h0'))
def test_ksc_rw_h0(case):
    check(bvt.sv.ksc_rw_h0, case)


@pytest.mark.parametrize('case', cases(FX, 'ksc_rw_diffuse'))
def test_ksc_rw_diffuse(case):
    h, S = run(bvt.sv.ksc_rw_diffuse, case)
    h_m, S_m = outputs(case)
    close(h, h_m)
    close(S + 1, S_m)                   # the mixture indicators count from 0


@pytest.mark.parametrize('case', cases(FX, 'nu_studentt'))
def test_nu_studentt(case):
    check(grid(bvt.sv.nu_studentt), case)
