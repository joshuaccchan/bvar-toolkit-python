# Does Modeling the Volatility Improve My Forecasts?

*Code: [`build.py`](build.py) and [`your_data.py`](your_data.py), which call
[`bvt.models.var_csv`](../../src/bvar_toolkit/models/_var_csv.py),
[`bvt.models.var_sv`](../../src/bvar_toolkit/models/_var_sv.py),
[`bvt.forecast.simulate`](../../src/bvar_toolkit/forecast/_simulate.py) and
[`bvt.forecast.mixquantile`](../../src/bvar_toolkit/forecast/_mixquantile.py). Method:
[Carriero, Clark and Marcellino (2016)](https://github.com/joshuaccchan/bvar-toolkit/blob/main/CITING.md#carriero-clark-and-marcellino-2016),
[Chan (2020)](https://github.com/joshuaccchan/bvar-toolkit/blob/main/CITING.md#chan-2020a) and
[Chan, Koop and Yu (2024)](https://github.com/joshuaccchan/bvar-toolkit/blob/main/CITING.md#chan-koop-and-yu-2024).
MATLAB version: [the same tutorial in bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/forecasting).*

In this tutorial we forecast the five quarterly US series of [the volatility-specification
tutorial](../sv_specification/), recursively and out of sample, from three reduced-form BVARs: a
homoskedastic one, one with a common volatility factor, and one with a volatility process per
equation. Each is re-estimated at every forecast origin and scored two ways: by the root mean
squared forecast error (RMSFE) of its point forecast, and by the log predictive likelihood of the
realized value under its whole predictive density.

Over 140 forecasts from 1990 both volatility models improve on the homoskedastic benchmark by both
criteria. The density gains are largest for the pandemic quarters and are not confined to them: 25
and 34 percent of the one-quarter-ahead gain and 48 and 72 percent of the four-quarter-ahead gain
come from quarters before 2020. The predictive intervals of a homoskedastic VAR are too wide in
calm years and far too narrow in a crisis.

## Try It Now

Five commands, from the root of the repository, after `pip install -e ".[tutorials]"`:

| Command | What it produces | Time |
|---|---|---|
| `python tutorials/forecasting/your_data.py` | the same comparison over a short window with short chains | about 80 seconds |
| `python tutorials/forecasting/forecast_now.py` | a forecast of the quarters after the sample, with intervals and an event probability | about 20 seconds |
| `python tutorials/forecasting/bench_density.py` | the two ways of scoring a multi-step density, on identical draws | about 40 seconds |
| `python tutorials/forecasting/check_stability.py` | the same quantities from chains of other seeds and four times the length | about seven minutes |
| `python tutorials/forecasting/build.py` | every table and figure of the forecast comparison | 41 minutes |

![Cumulative log score difference](fig_cumscore.png)

*Figure 1: The running sum of the joint log predictive likelihood of each volatility model minus
that of the homoskedastic VAR, against the quarter in which the forecast is made. A rising line
indicates the more accurate model. The top row uses every forecast; the bottom row keeps only
those whose target quarter falls before 2020, and the vertical scales differ across panels.*

## The Three Models

All three are reduced-form VARs with four lags. The first is homoskedastic, with
$`\mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})`$ under the natural conjugate
prior. The second is VAR-CSV,

```math
\mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \mathrm{e}^{h_t}\boldsymbol{\Sigma}), \qquad
h_t = \phi h_{t-1} + \varepsilon_t,
```

where the mean of $`h_t`$ is fixed at zero, so that $`\boldsymbol{\Sigma}`$ carries the scale of the
errors and one factor scales the whole matrix. This is the common stochastic volatility of
Carriero, Clark and Marcellino (2016), under the same prior as the first model, estimated by the
algorithm of Chan (2020), whose volatility step is the accept-reject Metropolis-Hastings of Chan
(2017). The third is VAR-OISV, with one log-volatility per equation and the order-invariant impact
matrix of Chan, Koop and Yu (2024), under that paper's Minnesota-type horseshoe prior, whose
own-lag and cross-lag shrinkage are estimated.

None of the three depends on the order of the columns, which is the subject of [the ordering
tutorial](../variable_ordering/). The first two share the natural conjugate prior, so the
difference between them is the volatility model alone. The third uses the Minnesota-type horseshoe
prior, so that comparison is between whole model-prior configurations.

Each model is estimated from 5,000 draws after a burn-in period of 1,000 and forecast one to four
quarters ahead. The log predictive likelihood of a realized value is the log of the average of its
predictive density over those draws.

## The Data and the Forecast Target

The five quarterly series are read from [`macro5_Q.csv`](../sv_specification/macro5_Q.csv) in the
folder of [the volatility-specification tutorial](../sv_specification/), 1971Q1 to 2024Q4:

| Series | Units |
|---|---|
| Unemployment rate | percent |
| PCE inflation | annualized quarterly rate, percent |
| Federal funds rate | percent |
| Chicago Fed National Financial Conditions Index | index, positive values indicating tighter conditions than average |
| Real GDP growth | annualized quarterly rate, percent |

With four lags and eight initial conditions, the first estimation sample ends in 1989Q4 and the
first forecast is of 1990Q1. Every model at every origin uses the data up to that quarter in the
current vintage, so the exercise is pseudo-out-of-sample.

## How the Densities Are Computed

For the homoskedastic VAR the $`h`$-step predictive distribution is Gaussian in closed form, and
[`bvt.forecast.predictive`](../../src/bvar_toolkit/forecast/_predictive.py) returns its mean and
variance for each draw. When the covariance varies over time no closed form is available, since
the covariance at $`T+h`$ is unknown at $`T`$, and the density must be simulated. Two estimators are
available.

The first simulates a whole path of the data for each draw and evaluates the realized value at the
state that path reaches. It is the noisier of the two, because few simulated paths land near an
extreme realized value, so the average that results is determined by a small number of draws.

`bvt.forecast.simulate` implements the second. Only the volatility path is simulated; given that
path the $`h`$-step distribution is Gaussian, with the mean iterating the VAR and the variance

```math
\mathbf{V}_h = \sum_{i=0}^{h-1} \boldsymbol{\Psi}_i \boldsymbol{\Sigma}_{T+h-i} \boldsymbol{\Psi}_i',
```

where $`\boldsymbol{\Psi}_i`$ are the moving-average matrices of the VAR. The density is therefore
evaluated exactly and only the volatility is integrated by Monte Carlo. The predictive
distribution that results is a mixture: conditioning on the coefficients and the volatility path
gives a normal, and the average of those normals over the draws has heavier tails than any one
of them.

[`bench_density.py`](bench_density.py) compares the two on identical posterior draws. Beyond one
quarter the path-based estimator returns a lower score and a wider spread across simulation seeds,
and both gaps are largest where the realized value is extreme: at the pandemic origin, four
quarters ahead, it gives −18.91 against −18.39, with seed ranges of 8.88 against 0.35. A
multi-step density score that changes materially between seeds usually has this cause.

As an implementation check the build also scores the homoskedastic model both ways, by simulation
and in closed form; the two agree to `0.0000` at both horizons.

## Accuracy Relative to the Homoskedastic VAR

There are 140 forecasts at one quarter ahead and 137 at four, since the last three origins
have no realized value four quarters ahead.

*Table 1: Accuracy relative to the homoskedastic VAR. The RMSFE column reports the median
percentage gain across the five variables; the log score column reports the mean gain in the joint
log predictive likelihood, per quarter.*

| Model | Horizon | Group | Forecasts | RMSFE gain | Log score gain |
|---|---|---|---|---|---|
| VAR-CSV | 1 | all | 140 | 5.0% | 5.39 |
| | 1 | targets through 2019 | 120 | 5.2% | 1.59 |
| | 1 | targets 2020 onwards | 20 | −1.0% | 28.16 |
| VAR-CSV | 4 | all | 137 | 4.9% | 2.52 |
| | 4 | targets through 2019 | 117 | 13.5% | 1.43 |
| | 4 | targets 2020 onwards | 20 | −0.0% | 8.91 |
| VAR-OISV | 1 | all | 140 | 8.7% | 5.52 |
| | 1 | targets through 2019 | 120 | 3.8% | 2.20 |
| | 1 | targets 2020 onwards | 20 | 22.6% | 25.42 |
| VAR-OISV | 4 | all | 137 | 17.6% | 1.90 |
| | 4 | targets through 2019 | 117 | 20.7% | 1.60 |
| | 4 | targets 2020 onwards | 20 | 8.4% | 3.66 |

The density gains are largest for pandemic targets: the mean gain for targets from 2020 is 18 and
12 times that for earlier targets at one quarter, for VAR-CSV and VAR-OISV, and 6 and 2 times at
four quarters. The gains are not confined to those quarters: multiplying each block's mean by its
count, the calm block accounts for 25 percent of VAR-CSV's total one-quarter gain and 34 percent
of VAR-OISV's, and 48 and 72 percent of the four-quarter gains.

The ranking of the two volatility models depends on the period. For targets through 2019 VAR-OISV
has the larger density gain at both horizons, 2.20 against 1.59 at one quarter and 1.60 against
1.43 at four, and it is ahead in 89 of those 120 quarters at one quarter and in 66 of the 117 at
four. For targets from 2020 VAR-CSV has the larger gain, 28.16 against 25.42 at one quarter and
8.91 against 3.66 at four. At one quarter that lead comes from a single quarter: VAR-CSV scores 81
log points above VAR-OISV for the 2020Q2 target and is 26 log points behind over the other
nineteen. At four quarters it is broader, 49 log points from the 2020Q2 target and 56 from the rest
of the block.

The predictive intervals account for the difference. One factor scales all five variances by the
same number, and it moves further in both directions than five separate volatilities do: the
VAR-CSV predictive standard deviation of GDP growth peaks at 26.9 for the forecast made in 2020Q2,
against VAR-OISV's 18.3 for the one made in 2020Q3, while its median through 2019 is 2.04 against
2.37. A wider band raises the predictive density at an extreme realized value and lowers it at an
ordinary one.

At four quarters both volatility models improve the point forecasts mostly in calm quarters: their
RMSFE gains are 13.5 and 20.7 percent for targets through 2019, against −0.0 and 8.4 percent for
targets from 2020. For pandemic targets VAR-CSV is no more accurate than the benchmark at either
horizon, and VAR-OISV improves on it by 22.6 percent at one quarter and 8.4 percent at four.

*Table 2: Accuracy by variable, against the homoskedastic VAR. The RMSFE column is the percentage
gain in root mean squared forecast error and the score column the mean gain in that variable's own
log predictive likelihood, so the five do not add up to the joint gain of Table 1.*

| Horizon | Variable | VAR-CSV RMSFE | VAR-CSV score | VAR-OISV RMSFE | VAR-OISV score |
|---|---|---|---|---|---|
| 1 | Unemployment | −0.2% | 3.47 | −5.8% | 3.06 |
| 1 | PCE inflation | 2.9% | 0.23 | 7.9% | 0.19 |
| 1 | Fed funds | 12.8% | 0.57 | 45.0% | 0.95 |
| 1 | NFCI | 19.4% | 0.63 | 38.3% | 0.99 |
| 1 | GDP growth | 5.0% | 0.54 | 8.7% | 0.53 |
| 4 | Unemployment | −3.1% | 0.50 | −10.3% | 0.56 |
| 4 | PCE inflation | 16.2% | 0.30 | 17.6% | 0.25 |
| 4 | Fed funds | 4.9% | 0.18 | 19.2% | 0.24 |
| 4 | NFCI | 21.4% | 0.46 | 31.7% | 0.63 |
| 4 | GDP growth | 2.6% | 0.57 | 3.4% | 0.48 |

The median reported in Table 1 summarizes a wide spread. The point-forecast gains come from the
financial and policy series: at one quarter, 12.8 and 45.0 percent for the federal funds rate and
19.4 and 38.3 percent for the NFCI, against 2.9 and 7.9 percent for inflation. Unemployment is the
one series both volatility models forecast less accurately than the benchmark, by 0.2 and 5.8
percent at one quarter and by 3.1 and 10.3 percent at four. Its density gain at one quarter, 3.47
for VAR-CSV, is the largest of any variable at either horizon. For that series the volatility
models improve the interval and leave the point forecast worse.

Over all forecasts VAR-OISV has the larger RMSFE gain at both horizons, and VAR-CSV the larger log
score gain at four quarters, 2.52 against 1.90. At one quarter the two log score gains, 5.39 and
5.52, differ by less than the seed range of the 2020Q2 score alone, 0.6 when averaged over the 140
forecasts.

## The Predictive Intervals

![Predictive standard deviation of GDP growth](fig_psd.png)

*Figure 2: One-quarter-ahead predictive standard deviation of GDP growth at each origin.*

*Table 3: Median one-quarter-ahead predictive standard deviation of GDP growth, by the quarter
forecast.*

| Model | Targets through 2019 | Targets 2020 onwards |
|---|---|---|
| Homoskedastic | 2.80 | 3.93 |
| VAR-CSV | 2.04 | 3.41 |
| VAR-OISV | 2.37 | 3.12 |

The homoskedastic VAR has the widest interval in both blocks. It fits one covariance matrix to
thirty years, so its density is too diffuse in calm quarters, which costs log points in most of
them, and far too tight in 2020, which costs a large number at once. Its interval does widen after
2020, from 2.80 to 3.93, because the pandemic observations enter the estimation sample and remain
in it.

*Table 4: Coverage of the predictive intervals for GDP growth and inflation, and their average
width, over all 140 forecasts at one quarter and 137 at four. A well calibrated 80 percent
interval covers 80 percent of the realized values.*

| Variable | Horizon | Model | 80% coverage | 80% width | 95% coverage | 95% width |
|---|---|---|---|---|---|---|
| GDP growth | 1 | Homoskedastic | 87% | 7.67 | 97% | 11.78 |
| GDP growth | 1 | VAR-CSV | 77% | 6.95 | 93% | 11.81 |
| GDP growth | 1 | VAR-OISV | 83% | 6.79 | 94% | 11.39 |
| PCE inflation | 1 | Homoskedastic | 87% | 3.84 | 95% | 5.88 |
| PCE inflation | 1 | VAR-CSV | 79% | 3.72 | 96% | 6.32 |
| PCE inflation | 1 | VAR-OISV | 84% | 3.38 | 96% | 5.55 |
| GDP growth | 4 | Homoskedastic | 86% | 8.40 | 96% | 12.90 |
| GDP growth | 4 | VAR-CSV | 82% | 8.00 | 96% | 14.76 |
| GDP growth | 4 | VAR-OISV | 83% | 7.07 | 95% | 12.21 |
| PCE inflation | 4 | Homoskedastic | 81% | 5.59 | 94% | 8.62 |
| PCE inflation | 4 | VAR-CSV | 84% | 5.22 | 96% | 9.38 |
| PCE inflation | 4 | VAR-OISV | 82% | 4.70 | 94% | 7.75 |

With 140 forecasts the standard error of a coverage rate near 80 percent is about 3 percentage
points, so differences of a few points carry little information, and the forecasts are not
independent. One comparison is larger than that. The homoskedastic 80 percent interval for GDP
growth covers 87 percent of the realized values at one quarter, two standard errors above its
nominal level, and it is also the widest of the three, 7.67 against 6.79 for VAR-OISV. Both
volatility models reach a coverage nearer the nominal level with a narrower interval, which is
what the log scores in Table 1 reward.

## Forecasts Beyond the End of the Sample

The exercise above stops before the last observation, so that every forecast has a realized value
to be evaluated against. [`forecast_now.py`](forecast_now.py) estimates all three models through
the final quarter of the sample and reports their forecasts for the quarters ahead.

![GDP growth forecast](fig_forecast_now.png)

*Figure 3: Forecast of GDP growth for 2025 from VAR-CSV, estimated through 2024Q4. The dashed line
denotes the posterior median and the shaded areas the 68 and 90 percent predictive intervals.*

[`bvt.forecast.mixquantile`](../../src/bvar_toolkit/forecast/_mixquantile.py) inverts the
predictive mixture for the intervals, and the probability of an event is its cdf at the threshold:

```python
p = np.mean(ndtr((0 - mu) / sd))      # predictive probability of a fall below zero
```

where `ndtr` is the normal cdf of `scipy.special`, and `mu` and `sd` hold, for each draw, the mean
and the standard deviation that `bvt.forecast.simulate` returns for that variable and horizon.
Fitting one normal to the mixture's mean and variance, or taking quantiles of the draws' means,
would both understate the uncertainty.

*Table 5: Probability that quarterly GDP growth is negative, estimated through 2024Q4. The target
is the annualized growth rate of each quarter.*

| Model | 2025Q1 | 2025Q2 | 2025Q3 | 2025Q4 |
|---|---|---|---|---|
| Homoskedastic | **32%** | 28% | 29% | 31% |
| VAR-CSV | **5%** | 6% | 7% | 9% |
| VAR-OISV | 9% | 12% | 12% | 12% |

The medians for 2025Q1 are 1.7, 2.9 and 3.0 percent, and the intervals differ far more: the
homoskedastic 90 percent interval for GDP growth is [−4.5, 7.9] against VAR-CSV's [−0.0, 5.8],
because a single covariance matrix fitted to 1971 through 2024 still reflects 2020 five years
later.

The script writes the forecasts and the event probabilities to `outdir`, which defaults to the
temporary directory. The event and the variable drawn in the fan chart are settings at the top of
it.

## Checking the Estimates

Both chains are re-run at every origin, so their mixing matters. At the last origin, from 5,000
draws with a bandwidth of 200, the inefficiency factors are 11, 19 and 76 for VAR-CSV's
persistence, its volatility variance and $`\Sigma_{11}`$, and 22, 12 and 24 for VAR-OISV's first
persistence, its first volatility variance and the log-volatility at the end of the sample.
Section 6.5 of *Bayesian Macroeconometrics* sets out the diagnostic, and
[ex13 of bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit/blob/main/examples/ex13_mcmc_diagnostics.m)
applies it in detail.

An inefficiency factor describes the draws at one origin. Whether the numbers on this page would
come out the same from another chain is a separate question, and
[`check_stability.py`](check_stability.py) measures it: at four origins, two ordinary, one just
before the pandemic and one inside it, each volatility model is estimated three times at the
build's settings under different seeds and once at four times the length. The three seeds spread
the joint log score by at most 0.55 log points at one quarter and 0.69 at four, a chain of 20,000
draws differs from one of 5,000 at the same seed by at most 0.48, and the predictive probability
that GDP growth is negative moves by at most half a percentage point.

## Applying the Method to Your Data

The script [`your_data.py`](your_data.py) runs the same comparison on any dataset. Set the file,
the columns, the lag length, the evaluation window and the chain lengths in its settings block. The
loop is three calls per origin:

```python
res = bvt.models.var_csv(Y0, Y, p, nsim=nsim, burnin=burnin, draws=True)
dr = {'A': res.draws.A[d].reshape((k, n), order='F'),
      'Sig': res.draws.Sig[d].reshape((n, n), order='F'), 'h_T': res.draws.h[d, -1],
      'phi': res.draws.phi[d], 'sigh2': res.draws.sigh2[d]}
yhat, lden, ljoint, sdev = bvt.forecast.simulate('csv', dr, cfg)
```

where `cfg` holds the last `p` observations, the longest horizon and the realized value.
`bvt.models.var_sv` returns the same shape of draws for the order-invariant model, with `impact`,
`h_T`, `phi` and `sig2` in place of the common factor's fields, and `simulate` takes `'oisv'` for
it. The script writes the scores to `outdir`, which defaults to the temporary directory.

## Reproducing the Results

To reproduce the forecast comparison, run the build script from the root of the repository:

```
python tutorials/forecasting/build.py
```

The computation takes 41.4 minutes using Python 3.13 with NumPy 2.5 and SciPy 1.18 on a computer
with an Intel Core Ultra 7 255U processor and 32 GB of RAM. The script prints Tables 1 to 4, the
simulation check and the inefficiency factors, and saves Figures 1 and 2 in the same folder. It
also saves the scores of every individual forecast, with the quarter in which it is made and the
quarter it is for, in `scores_by_origin.npz`; the other numbers of the comparison are computed
from that file, and a regrouping or a question about a single quarter requires no rerun. Table 5
and Figure 3 come from [`forecast_now.py`](forecast_now.py), and the numbers of the density
benchmark and of the stability check from [`bench_density.py`](bench_density.py) and
[`check_stability.py`](check_stability.py); none of the three is part of the build.

The models are those of the MATLAB page and the draws come from NumPy's random numbers, so every
number that involves posterior draws differs from the MATLAB page within Monte Carlo error.
[`PARITY.md`](../../PARITY.md) sets the two side by side.

## References

Carriero, A., Clark, T. E. and Marcellino, M. (2016). Common drifting volatility in large Bayesian
VARs. *Journal of Business and Economic Statistics*, 34(3): 375-390.

Chan, J. C. C. (2017). The stochastic volatility in mean model with time-varying parameters: an
application to inflation modeling. *Journal of Business and Economic Statistics*, 35(1): 17-28.

Chan, J. C. C. (2020). Large Bayesian VARs: a flexible Kronecker error covariance structure.
*Journal of Business and Economic Statistics*, 38(1): 68-79.

Chan, J. C. C., Koop, G. and Yu, X. (2024). Large order-invariant Bayesian VARs with stochastic
volatility. *Journal of Business and Economic Statistics*, 42(2): 825-837.
