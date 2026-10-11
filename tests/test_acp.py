"""The functions behind the asymmetric conjugate prior: priors, closed-form marginal
likelihood, independent posterior draws, the map to the reduced form and the predictive
moments."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

PR = fixtures('priors')
ML = fixtures('ml')
SA = fixtures('samplers')
ST = fixtures('structural')
FC = fixtures('forecast')


def zero_based(x):
    """Positions from MATLAB, counted from 1, as positions counted from 0."""
    return (np.atleast_1d(np.asarray(x, dtype=int)) - 1).tolist()


def check_prior(py, ml):
    assert set(py) == set(ml)
    for key in ml:
        close(py[key], ml[key])


def prior_args(args, kwargs):
    args[0], args[1] = int(args[0]), int(args[1])
    if len(args) > 4:
        args[4] = zero_based(args[4])
    return args, kwargs


@pytest.mark.parametrize('name,case', [(n, c) for n in ['acp_stru', 'acp_redu'] for c in cases(PR, n)])
def test_priors(name, case):
    out = run(getattr(bvt.priors, name), case, prior_args)
    check_prior(out[0], outputs(case)[0])


@pytest.mark.parametrize('case', cases(PR, 'resid_var_ar4'))
def test_resid_var_ar4(case):
    check(bvt.priors.resid_var_ar4, case)


@pytest.mark.parametrize('case', cases(PR, 'acp_opt_kappa'))
def test_acp_opt_kappa(case):
    def adapt(args, kwargs):
        args[3] = int(args[3])
        args[6] = zero_based(args[6]) if np.size(args[6]) else None
        return args, kwargs
    check(bvt.priors.acp_opt_kappa, case, adapt)


@pytest.mark.parametrize('case', cases(ML, 'acp'))
def test_ml_acp(case):
    def adapt(args, kwargs):
        args[0] = int(args[0])
        return args, kwargs
    check(bvt.ml.acp, case, adapt)


@pytest.mark.parametrize('case', cases(SA, 'acp_theta_sig'))
def test_acp_theta_sig(case):
    def adapt(args, kwargs):
        args[2], args[4] = int(args[2]), int(args[4])
        return args, kwargs
    check(bvt.samplers.acp_theta_sig, case, adapt)


@pytest.mark.parametrize('case', cases(ST, 'reduced_form'))
def test_reduced_form(case):
    check(bvt.structural.reduced_form, case)


@pytest.mark.parametrize('case', cases(FC, 'predictive'))
def test_predictive(case):
    def adapt(args, kwargs):
        args[3] = int(args[3])
        if len(args) > 4:
            args[4] = zero_based(args[4])
        if np.ndim(args[2]) == 1:               # loadmat squeezes an n x 1 ylag
            args[2] = np.asarray(args[2])[:, None]
        return args, kwargs
    check(bvt.forecast.predictive, case, adapt)


def test_predictive_matches_companion_form():
    """One draw: the h-step variance equals the companion-form recursion."""
    rng = np.random.default_rng(1)
    n, p, H = 3, 2, 6
    Bt, St = bvt.structural.reduced_form(.3 * rng.standard_normal((1, 3)),
                                         .2 * rng.standard_normal((1, n * n * p + n)),
                                         rng.random((1, n)) + .5)
    ylag = rng.standard_normal((n, p))
    mu, sd = bvt.forecast.predictive(Bt, St, ylag, H)
    A = Bt[0].reshape(n, n * p + 1).T                # (k, n)
    F = np.zeros((n * p, n * p))
    F[:n] = A[1:].T
    F[n:, :-n] = np.eye(n * (p - 1))
    Q = np.zeros((n * p, n * p))
    Q[:n, :n] = St[0]
    V = np.zeros((n * p, n * p))
    for h in range(H):
        V = F @ V @ F.T + Q
        assert np.allclose(sd[0, :, h], np.sqrt(np.diag(V)[:n]))
