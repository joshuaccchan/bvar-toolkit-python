"""The marginal likelihoods of Chan (2023) for the five volatility specifications: the
densities and the importance density they use, the priors and sampler blocks of the models,
and the four importance-sampling estimators, compared with MATLAB draw for draw."""

import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

ML = fixtures('ml')
PR = fixtures('priors')
SA = fixtures('samplers')
SV = fixtures('sv')


@pytest.mark.parametrize('name', ['lgampdf', 'lmvnpdf_pcn', 'ltnormpdf'])
def test_densities(name):
    for case in cases(ML, name):
        check(getattr(bvt.ml, name), case)


def test_isden_arss():
    for case in cases(ML, 'isden_arss'):
        check(bvt.ml.isden_arss, case)


@pytest.mark.parametrize('case', cases(PR, 'minn'))
def test_minn(case):
    def adapt(args, kwargs):
        args[0], args[6] = int(args[0]), int(args[6])
        return args, kwargs
    check(bvt.priors.minn, case, adapt)


@pytest.mark.parametrize('case', cases(PR, 'impact_B0'))
def test_impact_B0(case):
    check(bvt.priors.impact_B0, case)


@pytest.mark.parametrize('name', ['eq_var_redu_tri', 'factor_fsv', 'eq_fsv_load'])
def test_samplers(name):
    for case in cases(SA, name):
        check(getattr(bvt.samplers, name), case)


@pytest.mark.parametrize('name', ['init_approx1N', 'svo_outlier'])
def test_sv(name):
    for case in cases(SV, name):
        check(getattr(bvt.sv, name), case)


@pytest.mark.parametrize('name, case', [(nm, c) for nm in ('mlvarsv_csv', 'mlvarsv_arsv_redu',
                                                          'mlvarsv_arsvo_redu', 'mlvarsv_fsv')
                                        for c in cases(ML, nm)],
                         ids=lambda v: v if isinstance(v, str) else str(v['seed']))
def test_mlvarsv(name, case):
    # VAR-SVO fits a beta density to the draws of po by fminsearch, which stops within 1e-6 of
    # the optimum; exp differs between NumPy and MATLAB in its last digit, and the fit ends 2e-7
    # away from MATLAB's, which moves the weights by 3e-9
    tol = 1e-7 if name == 'mlvarsv_arsvo_redu' else 1e-10
    lml, lmlstd, out = run(getattr(bvt.ml, name), case)
    lml_m, lmlstd_m, out_m = outputs(case)
    close(out.store_w, out_m['store_w'], tol)
    close(out.bigml, out_m['bigml'], tol)
    close(lml, lml_m, tol)
    close(lmlstd, lmlstd_m, tol * 1e3)
