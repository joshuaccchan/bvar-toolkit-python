"""The VAR with stochastic volatility under the Cholesky and the order-invariant
specifications: its prior building blocks, each block of the sampler and short chains of the
complete sampler, compared with MATLAB draw for draw."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

PR = fixtures('priors')
SA = fixtures('samplers')
SV = fixtures('sv')
ST = fixtures('structural')
MO = fixtures('models')


def from_one(*positions):
    """An adapt function that counts the given arguments from 0."""
    def adapt(args, kwargs):
        for i in positions:
            args[i] = np.atleast_1d(np.asarray(args[i], dtype=int)) - 1
        return args, kwargs
    return adapt


@pytest.mark.parametrize('case', cases(PR, 'minnesota_C'))
def test_minnesota_C(case):
    def adapt(args, kwargs):
        return [int(args[0]), int(args[1]), args[2]], kwargs
    C, i1, i2 = run(bvt.priors.minnesota_C, case, adapt)
    C_m, i1_m, i2_m = outputs(case)
    close(C, C_m)
    close(i1 + 1, i1_m)
    close(i2 + 1, i2_m)


@pytest.mark.parametrize('case', cases(PR, 'vtheta'))
def test_vtheta(case):
    check(bvt.priors.vtheta, case, from_one(0, 1))


@pytest.mark.parametrize('case', cases(ST, 'construct_Sigt'))
def test_construct_Sigt(case):
    check(bvt.structural.construct_Sigt, case)


@pytest.mark.parametrize('case', cases(ST, 'b0_row_sampler'))
def test_b0_row_sampler(case):
    check(bvt.structural.b0_row_sampler, case)


@pytest.mark.parametrize('case', cases(SA, 'eq_var_oi'))
def test_eq_var_oi(case):
    check(bvt.samplers.eq_var_oi, case)


@pytest.mark.parametrize('case', cases(SA, 'eq_tri_cs'))
def test_eq_tri_cs(case):
    check(bvt.samplers.eq_tri_cs, case)


@pytest.mark.parametrize('case', cases(SA, 'alp_tri_cs'))
def test_alp_tri_cs(case):
    check(bvt.samplers.alp_tri_cs, case)


@pytest.mark.parametrize('case', cases(SA, 'horseshoe_kappa_psi'))
def test_horseshoe_kappa_psi(case):
    check(bvt.samplers.horseshoe_kappa_psi, case, from_one(1, 2))


@pytest.mark.parametrize('case', cases(SV, 'ksc_ar1_mean'))
def test_ksc_ar1_mean(case):
    h, S = run(bvt.sv.ksc_ar1_mean, case)
    h_m, S_m = outputs(case)
    close(h, h_m)
    close(S + 1, S_m)                              # MATLAB counts the components from 1


@pytest.mark.parametrize('case', cases(SV, 'sv_params'))
def test_sv_params(case):
    check(bvt.sv.sv_params, case)


@pytest.mark.parametrize('case', cases(MO, 'var_sv'))
def test_var_sv(case):
    def adapt(args, kwargs):
        args[2] = int(args[2])
        return args, kwargs
    res, = run(bvt.models.var_sv, case, adapt)
    ml, = outputs(case)
    for f in ('Sig_mean', 'A_mean', 'h_mean', 'impact_mean', 'kappa_mean'):
        close(res[f], ml[f])
    if 'draws' in ml:
        for f in ml['draws']:
            close(res.draws[f], ml['draws'][f])
