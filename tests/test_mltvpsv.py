"""The marginal likelihoods of Chan and Eisenstat (2018) for VARs with time-varying
coefficients, stochastic volatility or Markov switching: the Dirichlet density and its fit,
the integrated likelihoods, and the ten importance sampling estimators, compared with MATLAB
draw for draw. The log priors are those of export_fixtures.m, with the same order of
operations."""

import numpy as np
import pytest

import bvar_toolkit as bvt
from conftest import cases, check, close, fixtures, outputs, run

ML = fixtures('ml')


def lpt(th):
    return -.5 * np.sum(th**2) / 10


def lps(v):
    return -np.sum(2 * np.log(v) + .1 / v)


def lpP(P):
    return np.log(P[0, 0]) + np.log(P[1, 1])


def lp_rs(th, sg, P):
    return lpt(th) + lps(sg) + lpP(P)


# the position of the prior among the arguments, and the prior
PRIORS = {
    'mltvpsv_cvar': (3, lambda th, sg: lpt(th) + lps(sg)),
    'mltvpsv_cvarsv': (4, lambda th, sh, h0: lpt(th) + lps(sh) + lpt(h0)),
    'mltvpsv_tvp': (4, lambda sg, st, t0: lps(sg) + lps(st) + lpt(t0)),
    'mltvpsv_tvpsv': (5, lambda st, sh, t0, h0: lps(st) + lps(sh) + lpt(t0) + lpt(h0)),
    'mltvpsv_rs': (4, lp_rs),
    'mltvpsv_rs_r1': (4, lp_rs),
    'mltvpsv_rs_r2': (4, lp_rs),
    'mltvpsv_tvp_r1_sv': (6, lambda b, sg, sh, g0, h0: lpt(b) + lps(sg) + lps(sh) + lpt(g0)
                          + lpt(h0)),
    'mltvpsv_tvp_r2_sv': (6, lambda g, sb, sh, b0, h0: lpt(g) + lps(sb) + lps(sh) + lpt(b0)
                          + lpt(h0)),
    'mltvpsv_tvp_r3_sv': (7, lambda b, g, sm, sh, m0, h0: lpt(b) + lpt(g) + lps(sm) + lps(sh)
                          + lpt(m0) + lpt(h0)),
}


@pytest.mark.parametrize('name', ['ldiripdf', 'dirifit', 'intlike_tvp', 'intlike_rsvar',
                                  'intlike_cvarsv', 'intlike_tvpsv'])
def test_densities(name):
    for case in cases(ML, name):
        check(getattr(bvt.ml, name), case)


@pytest.mark.parametrize('name, case', [(nm, c) for nm in PRIORS for c in cases(ML, nm)],
                         ids=lambda v: v if isinstance(v, str) else str(v['seed']))
def test_mltvpsv(name, case):
    pos, prior = PRIORS[name]

    def adapt(args, kwargs):
        args.insert(pos, prior)
        return args, kwargs

    lml, lmlstd, out = run(getattr(bvt.ml, name), case, adapt)
    lml_m, lmlstd_m, out_m = outputs(case)
    close(out.store_w, out_m['store_w'])
    close(out.bigml, out_m['bigml'])
    close(lml, lml_m)
    close(lmlstd, lmlstd_m, 1e-7)
