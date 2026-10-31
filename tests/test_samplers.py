"""Sampler blocks: the missing values of a VAR, hard and soft restrictions included, the joint
draw of the VAR coefficients, the structural-form coefficients equation by equation, the
order-invariant draw of the coefficients, the hierarchical shrinkage of Chan (2021), and the
hybrid TVP block and its prior scales, compared with MATLAB draw for draw."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, grid, outputs, run

FX = fixtures('samplers')
PR = fixtures('priors')


@pytest.mark.parametrize('case', cases(FX, 'missing_var'))
def test_missing_var(case):
    def adapt(args, kwargs):
        args[4] = int(args[4])
        if 'O' in kwargs:
            kwargs['o'] = kwargs.pop('O')
        if 'ndraws' in kwargs:
            kwargs['ndraws'] = int(kwargs['ndraws'])
        return args, kwargs
    ym, ymhat, S = run(bvt.samplers.missing_var, case, adapt)
    ym_m, ymhat_m, S_m = outputs(case)
    close(ym, ym_m)
    close(ymhat, ymhat_m)
    for f in ('So', 'Sm', 'yo'):
        close(S[f], S_m[f])


@pytest.mark.parametrize('case', cases(FX, 'var_coef_joint'))
def test_var_coef_joint(case):
    A, Ahat = run(bvt.samplers.var_coef_joint, case)
    A_m, Ahat_m = outputs(case)
    close(A, A_m)
    close(Ahat, Ahat_m)


@pytest.mark.parametrize('name', ['eq_gauss', 'eq_svar_oi'])
def test_equation_blocks(name):
    for case in cases(FX, name):
        check(getattr(bvt.samplers, name), case)


@pytest.mark.parametrize('case', cases(FX, 'gig_shrinkage'), ids=lambda c: c['in1'])
def test_gig_shrinkage(case):
    def adapt(args, kwargs):
        args[2] = np.asarray(args[2], dtype=int) - 1        # positions count from 0
        args[3] = np.asarray(args[3], dtype=int) - 1
        args[11], args[12] = int(args[11]), int(args[12])
        return args, kwargs
    check(bvt.samplers.gig_shrinkage, case, adapt)


@pytest.mark.parametrize('case', cases(FX, 'nu_psi_ng'))
def test_nu_psi_ng(case):
    check(grid(bvt.samplers.nu_psi_ng), case)


@pytest.mark.parametrize('case', cases(FX, 'eq_hyb_tvp'))
def test_eq_hyb_tvp(case):
    def adapt(args, kwargs):
        args[6], args[7] = int(args[6]), bool(args[7])
        return args, kwargs
    check(bvt.samplers.eq_hyb_tvp, case, adapt)


@pytest.mark.parametrize('case', cases(PR, 'resid_var_allvars_ridge'))
def test_resid_var_allvars_ridge(case):
    check(bvt.priors.resid_var_allvars_ridge, case)
