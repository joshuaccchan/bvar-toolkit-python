import numpy as np

from .._struct import Struct


def tables(action, *args):
    """bvt.forecast.tables - forecast-evaluation accumulation and RMSFE / average
    log predictive likelihood (ALPL) table construction. Three named actions:

      row = bvt.forecast.tables('accum_row', tmpyhat, obs)
        One accumulation row for one vintage and one horizon: the realized
        value, the posterior mean point forecasts, and the log predictive
        likelihoods averaged over draws by log-sum-exp. tmpyhat is the nsim x
        (2n+1) per-draw matrix from bvt.forecast.iterate ([point forecasts,
        per-variable log prelikes, joint log prelike]); obs is the n-vector of
        realized values.
        The storage GUARDS (springer: store yhat1 only when t<=T-1; MAHP: yhat4
        only when t<=T-4) are loop control and stay with the CALLER.

      S = bvt.forecast.tables('springer', yhat0, yhat1, var_core)
        RMSFE / ALPL tables for the h=0 (nowcast) and h=1 accumulators. yhat0 and
        yhat1 are the vintage-by-row stacks of 'accum_row' output. The
        evaluation-period row trims are applied here: yhat0[4:] (1975Q1 on)
        and yhat1[3:]. A nonempty var_core (the positions of the headline columns,
        counted from 0, e.g. [0, 6, 7, 11]) tabulates those variables only, with no
        joint column; an empty var_core gives the all-variable (BVAR-small) form,
        per-variable AND joint. S has fields RMSFE_0, RMSFE_1, RMSFE = [RMSFE_0 RMSFE_1],
        aveprelike_0, aveprelike_1, aveprelike.

      S = bvt.forecast.tables('mahp', yhat1, yhat4)
        RMSFE / ALPL tables for the h=1 and h=4 accumulators, again stacks of
        'accum_row' output. Row trims yhat1[3:] and yhat4[0:], all n
        variables, per-variable AND joint. S has fields RMSFE_1, RMSFE_4, RMSFE,
        aveprelike_1, aveprelike_4, aveprelike.

    See:
    Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Ed.),
    Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham.
    Chan, J.C.C. (2021). Minnesota-Type Adaptive Hierarchical Priors for
    Large Bayesian VARs, International Journal of Forecasting, 37(3), 1212-1226.
    """
    if action == 'accum_row':
        tmpyhat = np.asarray(args[0], dtype=float)
        obs = np.asarray(args[1], dtype=float).ravel()
        n = (tmpyhat.shape[1] - 1) // 2
        tmpmax = np.max(tmpyhat[:, n:], axis=0)
        return np.r_[obs, np.mean(tmpyhat[:, :n], axis=0),
                     np.log(np.mean(np.exp(tmpyhat[:, n:] - tmpmax), axis=0)) + tmpmax]

    if action == 'springer':
        yhat0 = np.asarray(args[0], dtype=float)
        yhat1 = np.asarray(args[1], dtype=float)
        var_core = np.asarray(args[2], dtype=int).ravel() if args[2] is not None else []
        n = (yhat0.shape[1] - 1) // 3
        if len(var_core) == 0:           # model 1 (BVAR-small): all n variables
            RMSFE_0 = np.sqrt(np.mean((yhat0[4:, :n] - yhat0[4:, n:2 * n])**2, axis=0))
            RMSFE_1 = np.sqrt(np.mean((yhat1[3:, :n] - yhat1[3:, n:2 * n])**2, axis=0))
            aveprelike_0 = np.mean(yhat0[4:, 2 * n:], axis=0)
            aveprelike_1 = np.mean(yhat1[3:, 2 * n:], axis=0)
        else:                            # models 2-8: headline var_core columns
            RMSFE_0 = np.sqrt(np.mean((yhat0[4:, var_core] - yhat0[4:, n + var_core])**2, axis=0))
            RMSFE_1 = np.sqrt(np.mean((yhat1[3:, var_core] - yhat1[3:, n + var_core])**2, axis=0))
            aveprelike_0 = np.mean(yhat0[4:, 2 * n + var_core], axis=0)
            aveprelike_1 = np.mean(yhat1[3:, 2 * n + var_core], axis=0)
        return Struct(RMSFE_0=RMSFE_0, RMSFE_1=RMSFE_1,
                      RMSFE=np.column_stack([RMSFE_0, RMSFE_1]),
                      aveprelike_0=aveprelike_0, aveprelike_1=aveprelike_1,
                      aveprelike=np.column_stack([aveprelike_0, aveprelike_1]))

    if action == 'mahp':
        yhat1 = np.asarray(args[0], dtype=float)
        yhat4 = np.asarray(args[1], dtype=float)
        n = (yhat1.shape[1] - 1) // 3
        RMSFE_1 = np.sqrt(np.mean((yhat1[3:, :n] - yhat1[3:, n:2 * n])**2, axis=0))
        RMSFE_4 = np.sqrt(np.mean((yhat4[:, :n] - yhat4[:, n:2 * n])**2, axis=0))
        aveprelike_1 = np.mean(yhat1[3:, 2 * n:], axis=0)
        aveprelike_4 = np.mean(yhat4[:, 2 * n:], axis=0)
        return Struct(RMSFE_1=RMSFE_1, RMSFE_4=RMSFE_4,
                      RMSFE=np.column_stack([RMSFE_1, RMSFE_4]),
                      aveprelike_1=aveprelike_1, aveprelike_4=aveprelike_4,
                      aveprelike=np.column_stack([aveprelike_1, aveprelike_4]))

    raise ValueError(f"unknown action '{action}'; use accum_row, springer or mahp")
