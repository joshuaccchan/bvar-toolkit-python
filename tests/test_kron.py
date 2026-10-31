"""The marginal likelihoods of Chan (2020) for BVARs with a Kronecker error covariance: the
densities and likelihoods they evaluate, the integrated likelihoods by importance sampling,
and the eight estimators, compared with MATLAB draw for draw. In the replay runs, MATLAB's
fminunc is replaced by the Nelder-Mead search of the Python port (tools/matlab/replay)."""

import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

ML = fixtures('ml')
KRON = ['kron_bvar', 'kron_bvar_t', 'kron_bvar_ma', 'kron_bvar_csv', 'kron_bvar_t_csv',
        'kron_bvar_t_ma', 'kron_bvar_csv_ma', 'kron_bvar_csv_t_ma']


@pytest.mark.parametrize('name', ['linvgammpdf', 'lniwpdf', 'llike_ma', 'llike_csv_ma',
                                  'intlike_csv', 'intlike_csv_ma', 'intlike_t_csv',
                                  'intlike_csv_t_ma'])
def test_likelihoods(name):
    for case in cases(ML, name):
        check(getattr(bvt.ml, name), case)


@pytest.mark.parametrize('name, case', [(nm, c) for nm in KRON for c in cases(ML, nm)],
                         ids=lambda v: v if isinstance(v, str) else '')
def test_kron(name, case):
    # the MA estimators find the mode of the psi target by the Nelder-Mead search that stands
    # in for fminunc, at a tolerance of 1e-10. The rounding of the target differs between
    # NumPy and MATLAB by about 1e-14, which fixes the minimizer only to about 2e-8, and the
    # ordinates of the reduced runs end 5e-8 apart
    tol = 1e-6 if name in ('kron_bvar_t_ma', 'kron_bvar_csv_ma', 'kron_bvar_csv_t_ma') else 1e-10
    lml, out = run(getattr(bvt.ml, name), case)
    lml_m, out_m = outputs(case)
    close(lml, lml_m, tol)
    for field, value in out_m.items():
        close(out[field], value, tol)
