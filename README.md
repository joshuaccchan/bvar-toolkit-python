# bvar-toolkit for Python

[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) is a MATLAB library for large
Bayesian VARs: samplers, shrinkage priors, stochastic volatility, marginal likelihoods and
forecasting, built from fourteen published replication packages. This repository is its Python
version: the same functions under the same names, each tested against its MATLAB twin, and the
toolkit's tutorials with Python code.

It covers 117 of the 118 core functions and all six tutorials; `bvar.util.simulate_var`, which
the MATLAB how-to guides use, is not ported yet. [PARITY.md](PARITY.md) lists every function
and how it was checked.

## Installation

From a clone of this repository:

```
pip install -e ".[tutorials]"
```

The package needs NumPy, SciPy, pandas and matplotlib. The `tutorials` extra adds openpyxl,
which the shrinkage tutorial uses to read an Excel file.

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
[shrinkage tutorial](tutorials/shrinkage/) takes the same steps for 21 variables.

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
[`tools/matlab/export_fixtures.m`](tools/matlab/export_fixtures.m) runs each ported function in
MATLAB, at a pinned commit of bvar-toolkit, and saves its inputs and outputs in
`tests/fixtures`. The tests run the Python function on the same inputs and require every entry
of every output to agree to 1e-10. The largest difference measured is 2e-11. Four marginal
likelihoods are the exceptions, and PARITY.md explains them. That of VAR-SVO,
`bvt.ml.mlvarsv_arsvo_redu`, agrees to 4e-9, because a beta fit inside it stops within its
tolerance. Those of the three MA models of Chan (2020) agree to 6e-8: they search for the mode of
a density whose rounding differs between the languages, and the rounding fixes that mode only to
about 2e-8.

Functions that draw random numbers are compared on a shared stream. NumPy's `RandomState`
reproduces the uniforms of MATLAB's Mersenne Twister bit for bit, and both languages turn them
into normal, gamma and beta draws by the inverse distribution functions, into integers by
rounding up, into permutations by sorting and into inverse-Wishart draws by the Bartlett
decomposition: in MATLAB through the test versions of `rand`, `randn`, `gamrnd`, `betarnd`,
`unidrnd`, `randi`, `randperm` and `iwishrnd` in `tools/matlab/replay`, and in Python through
`bvt.Replay`. On that stream every draw agrees with MATLAB's, and so does the number of uniforms
used. A generalized inverse Gaussian draw returns the mode of its distribution in both test
versions. The optimizers `fminbnd` and `fminsearch` and the root finder `fzero` follow MATLAB's
algorithms and stopping rules, and so do the gamma and beta fits of the marginal likelihoods,
which use them. The line search of `fminunc` is not published: in its place, `fminsearch` finds
the minimum, and the second derivative comes from `fminunc`'s finite-difference formula. On the
test objectives this minimizer lies within 2e-8 of `fminunc`'s, and the test version of
`fminunc` in `tools/matlab/replay` runs the same search. On the test objectives the optimizers and the root finder return MATLAB's
results to the last digit and the fits agree to 5e-15, and [PARITY.md](PARITY.md) shows how
closely the optimizers track MATLAB on the marginal likelihood of the shrinkage tutorial.

The tests run on every push, together with the `your_data.py` script of every tutorial.

## Tutorials

| Question | MATLAB version |
|---|---|
| [Does the Order of the Variables Change My VAR Results?](tutorials/variable_ordering/) | [variable_ordering](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/variable_ordering) |
| [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](tutorials/shrinkage/) | [shrinkage](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/shrinkage) |
| [Which Stochastic Volatility Specification Should My VAR Use?](tutorials/sv_specification/) | [sv_specification](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/sv_specification) |
| [Does Modeling the Volatility Improve My Forecasts?](tutorials/forecasting/) | [forecasting](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/forecasting) |
| [Can I Use Many Sign Restrictions in My Large VAR?](tutorials/sign_restrictions/) | [sign_restrictions](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/sign_restrictions) |
| [How Do I Handle Missing and Mixed-Frequency Data in My VAR?](tutorials/mixed_frequency/) | [mixed_frequency](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/mixed_frequency) |

## Citation

The MATLAB toolkit's [CITING.md](https://github.com/joshuaccchan/bvar-toolkit/blob/main/CITING.md)
names the paper behind each function, with BibTeX entries. Please cite the papers whose
methods you use.

## License

MIT; see [LICENSE](LICENSE).
