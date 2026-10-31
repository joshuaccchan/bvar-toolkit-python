"""The functions behind sign and ranking restrictions: the sign-normalized QR, the two
acceptance rules, the separability check and the impulse responses."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

FX = fixtures('structural')


@pytest.mark.parametrize('case', cases(FX, 'qr_sign'))
def test_qr_sign(case):
    check(bvt.structural.qr_sign, case)


@pytest.mark.parametrize('case', cases(FX, 'irf_redu'))
def test_irf_redu(case):
    def adapt(args, kwargs):
        args[2], args[3] = int(args[2]), int(args[3])
        return args, kwargs
    check(bvt.structural.irf_redu, case, adapt)


@pytest.mark.parametrize('case', cases(FX, 'check_separable'))
def test_check_separable(case):
    ok, bad = run(bvt.structural.check_separable, case)
    ok_m, bad_m = outputs(case)
    assert ok == bool(ok_m)
    close(bad, np.asarray(bad_m, dtype=float).reshape(-1, 2) - 1)   # pairs count from 0 here


@pytest.mark.parametrize('case', cases(FX, 'sign_restrict'))
def test_sign_restrict(case):
    def adapt(args, kwargs):
        args[3] = np.atleast_1d(np.asarray(args[3], dtype=int)) - 1   # columns count from 0
        return args, kwargs
    ok, L = run(bvt.structural.sign_restrict, case, adapt)
    ok_m, L_m = outputs(case)
    assert ok == bool(ok_m)
    close(L, L_m)


@pytest.mark.parametrize('case', cases(FX, 'sign_assign'))
def test_sign_assign(case):
    def adapt(args, kwargs):
        args[3] = int(args[3])
        return args, kwargs
    ok, L = run(bvt.structural.sign_assign, case, adapt)
    ok_m, L_m = outputs(case)
    assert ok == bool(ok_m)
    close(L, L_m)


def test_sign_assign_rejects_unseparated_restrictions():
    """Two shocks with identical restrictions can draw the same column; the function then
    raises an error."""
    rng = np.random.default_rng(0)
    S = np.full((4, 2), np.nan)
    S[0, :] = 1
    for _ in range(200):
        L = rng.standard_normal((4, 4))
        try:
            bvt.structural.sign_assign(L, S, np.zeros((2, 4)), 1, rng=rng)
        except ValueError:
            return
    pytest.fail('no draw of the same column in 200 candidates')
