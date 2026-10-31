# How-To Guides

Each guide shows how to use the toolkit's core functions on one small example with generated data.
A guide is a page, `README.md`, and a script, `guide.py`, that prints the numbers and draws the
figures on the page. The guides come in series, and within a series each guide builds on the one
before. They are the Python versions of the
[guides of the MATLAB toolkit](https://github.com/joshuaccchan/bvar-toolkit/tree/main/guides).

## Getting Started with Bayesian VARs

The guides use the same generated VAR(2) of three variables.

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Stack the Equations of a VAR](var_stacking/) | Stack the equations in matrix form and as one long regression | `bvt.util.build_lags`, `bvt.util.surform2` |
| [How to Draw the VAR Coefficients under the Minnesota Prior](coefficient_draw/) | Get posterior draws of the VAR coefficients | `bvt.priors.minn`, `bvt.samplers.var_coef_joint` |
| [How to Estimate a VAR by Gibbs Sampling](gibbs_sampler/) | Draw the coefficients and the error covariance matrix in turn, and check the chain | `bvt.samplers.var_coef_joint`, `bvt.diag.inefficiency_factor`, `bvt.diag.mcse`, `bvt.diag.geweke` |
| [How to Choose the Shrinkage of a VAR by the Marginal Likelihood](natural_conjugate_ml/) | Choose the shrinkage and the lag length by the closed-form marginal likelihood | `bvt.util.ldet`, `bvt.util.mgammaln`, `bvt.ml.kron_bvar` |

## Forecasting with a VAR

The guides use data simulated from a VAR(4) fitted to US real GDP growth, PCE inflation and the
unemployment rate, 1960Q1 to 2019Q4.

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Simulate Data from an Estimated VAR](var_simulation/) | Simulate data from a VAR fitted to US GDP growth, inflation and unemployment | `bvt.util.build_lags`, `bvt.util.simulate_var` |
| [How to Forecast with a Bayesian VAR](var_forecast/) | Forecast from posterior draws, and compare with the true model's forecasts | `bvt.util.simulate_var`, `bvt.forecast.predictive`, `bvt.forecast.mixquantile` |
| [How to Evaluate Forecasts Out of Sample](var_forecast_evaluation/) | Score forecasts made at each quarter with the data of the time | `bvt.forecast.predictive`, `bvt.util.logsumexp` |
| [How to Choose the Lag Length and the Shrinkage for Forecasting](var_lag_shrinkage/) | Choose the lag length and the shrinkage by the marginal likelihood at each forecast origin | `bvt.util.build_lags`, `bvt.util.ldet`, `bvt.util.mgammaln` |

The [tutorials](../tutorials/) answer empirical questions with the settings of published papers,
and the [examples of the MATLAB toolkit](https://github.com/joshuaccchan/bvar-toolkit/tree/main/examples)
run the core functions on additional models, with real and generated data.
