# bvar-toolkit for Python

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23826635.svg)](https://doi.org/10.5281/zenodo.23826635)

[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) is a MATLAB library for large
Bayesian VARs: samplers, shrinkage priors, stochastic volatility, marginal likelihoods and
forecasting, built from fourteen published replication packages. This repository is its Python
version: the same functions under the same names, each tested against its MATLAB twin, and the
toolkit's tutorials and how-to guides with Python code.

It covers all 118 core functions, the six tutorials and the eight how-to guides.
[PARITY.md](https://github.com/joshuaccchan/bvar-toolkit-python/blob/main/PARITY.md) lists every function and how it was checked.

## Installation

```
pip install bvar-toolkit
```

installs the package with the NumPy, SciPy, pandas and matplotlib it needs. The tutorials and
the guides read the data in this repository, so to run them, install from a clone instead:

```
git clone https://github.com/joshuaccchan/bvar-toolkit-python.git
cd bvar-toolkit-python
pip install -e ".[tutorials]"
```

The `tutorials` extra adds openpyxl, which the shrinkage tutorial uses to read an Excel file.

## A First Example

The asymmetric conjugate prior of Chan (2022) shrinks the coefficients on a variable's own lags
and on other variables' lags by different amounts, and its marginal likelihood is available in
closed form. For a VAR of GDP growth, PCE inflation and the 3-month Treasury bill rate with
four lags, the two hyperparameters that maximize it take one call:

```python
import numpy as np
import bvar_toolkit as bvt

data = np.loadtxt('data/chan2019wp_acp/macrodata_Q_2018Q4.csv', delimiter=',')[:238, [0, 94, 144]]
p = 4
Y0, Y = data[:8], data[8:]                  # initial conditions and the estimation sample
_, Z = bvt.util.build_lags(np.vstack([Y0[-p:], Y]), p)
lml, kappa = bvt.priors.acp_opt_kappa(Y0, Y, Z, p, [.04, .04], 'stru')
```

Here `kappa[0]` and `kappa[1]` are the own-lag and other-lag hyperparameters, 0.542 and
0.0128, and `lml` is the maximized log marginal likelihood, -1237.5. The
[shrinkage tutorial](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/tutorials/shrinkage/) takes the same steps for 21 variables.

## Conventions

- **Names.** `bvar.util.build_lags` in MATLAB is `bvt.util.build_lags` here, after
  `import bvar_toolkit as bvt`, with the arguments in the same order. MATLAB's name-value
  options are keyword arguments.
- **Positions.** Positions of variables count from 0, as everywhere in Python.
- **Shapes.** Matrices keep their MATLAB shapes and vectors are 1-D arrays. `vec` stacks
  columns, as MATLAB's `Y(:)` does.
- **Structs.** A MATLAB struct is a `bvt.Struct`, a dict whose entries can also be read as
  attributes. Functions read structs by key, so a plain dict works as input.
- **Random numbers.** Every function that draws takes an `rng` argument: `None`, an integer
  seed or a `numpy.random.Generator`. It replaces the `'seed'` option of the MATLAB models. A
  seed gives different draws in Python and in MATLAB.

## Performance

NumPy and SciPy bundle OpenBLAS, which by default starts one thread per core. For the small
matrices of these samplers the threads can cost far more than they save: on the computer the
tutorials were timed on, an Intel Core Ultra 7 255U under Windows, solving a linear system of
539 equations took 440 milliseconds with the default threads and 3 milliseconds with one. Set
the environment variable `OPENBLAS_NUM_THREADS=1` before Python starts, as the tutorial scripts
do for themselves.

## How the Port Is Checked

The MATLAB toolkit is the reference.
[`tools/matlab/export_fixtures.m`](https://github.com/joshuaccchan/bvar-toolkit-python/blob/main/tools/matlab/export_fixtures.m) runs
each function in MATLAB, at a pinned commit of bvar-toolkit, and saves its inputs and outputs.
The tests run the Python function on the same inputs and require every entry of every output to
agree to 1e-10. Four marginal likelihoods, which end in a numerical search, are tested at looser
tolerances. Functions that draw random numbers run on a stream of uniforms that both languages
share, through `bvt.Replay` in Python and the test versions in `tools/matlab/replay` in MATLAB,
so every draw must agree. The optimizers and the root finder follow MATLAB's algorithms and
stopping rules. [PARITY.md](https://github.com/joshuaccchan/bvar-toolkit-python/blob/main/PARITY.md) gives the largest difference for every function
and explains the exceptions.

The tests run on every push, together with the `your_data.py` script of every tutorial and the
`guide.py` script of every guide.

## Tutorials

| Question | MATLAB version |
|---|---|
| [Does the Order of the Variables Change My VAR Results?](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/tutorials/variable_ordering/) | [variable_ordering](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/variable_ordering) |
| [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/tutorials/shrinkage/) | [shrinkage](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/shrinkage) |
| [Which Stochastic Volatility Specification Should My VAR Use?](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/tutorials/sv_specification/) | [sv_specification](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/sv_specification) |
| [Does Modeling the Volatility Improve My Forecasts?](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/tutorials/forecasting/) | [forecasting](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/forecasting) |
| [Can I Use Many Sign Restrictions in My Large VAR?](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/tutorials/sign_restrictions/) | [sign_restrictions](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/sign_restrictions) |
| [How Do I Handle Missing and Mixed-Frequency Data in My VAR?](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/tutorials/mixed_frequency/) | [mixed_frequency](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/mixed_frequency) |

## How-To Guides

Each [guide](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/) shows how to use the core functions on one small example with generated
data, in two series of four.

| Series | Guides |
|---|---|
| Getting Started with Bayesian VARs | [stacking the equations](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/var_stacking/), [drawing the coefficients](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/coefficient_draw/), [Gibbs sampling](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/gibbs_sampler/), [choosing the shrinkage by the marginal likelihood](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/natural_conjugate_ml/) |
| Forecasting with a VAR | [simulating data](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/var_simulation/), [forecasting](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/var_forecast/), [evaluating forecasts out of sample](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/var_forecast_evaluation/), [choosing the lag length and the shrinkage for forecasting](https://github.com/joshuaccchan/bvar-toolkit-python/tree/main/guides/var_lag_shrinkage/) |

## Citation

The MATLAB toolkit's [CITING.md](https://github.com/joshuaccchan/bvar-toolkit/blob/main/CITING.md)
names the paper behind each function, with BibTeX entries. Please cite the papers whose
methods you use.

## License

MIT; see [LICENSE](https://github.com/joshuaccchan/bvar-toolkit-python/blob/main/LICENSE).
