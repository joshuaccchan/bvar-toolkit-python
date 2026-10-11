# bvar-toolkit for Python

[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit) is a MATLAB library for large
Bayesian VARs: samplers, shrinkage priors, stochastic volatility, marginal likelihoods and
forecasting, built from fourteen published replication packages. This repository is its Python
version: the same functions under the same names, each tested against its MATLAB twin, and the
toolkit's tutorials with Python code.

The port is in progress. It covers 32 of the 117 core functions and one of the six tutorials
so far. [PARITY.md](PARITY.md) lists every function and how it was checked.

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
  seed or a `numpy.random.Generator`. A seed gives different draws in Python and in MATLAB.

## How the Port Is Checked

The MATLAB toolkit is the reference. [`tools/matlab/export_fixtures.m`](tools/matlab/export_fixtures.m)
runs each ported function in MATLAB, at a pinned commit of bvar-toolkit, and saves its inputs
and outputs in `tests/fixtures`. The tests run the Python function on the same inputs and
require every entry of every output to agree to 1e-10. The largest difference measured is
2e-13.

Functions that draw random numbers are compared on a shared stream. NumPy's `RandomState`
reproduces the uniforms of MATLAB's Mersenne Twister bit for bit, and both languages turn them
into normal, gamma and beta draws by the inverse distribution functions: in MATLAB through
the test versions of `rand`, `randn`, `gamrnd` and `betarnd` in `tools/matlab/replay`, and in
Python through `bvt.Replay`. On that stream every draw agrees with MATLAB's, and so does the
number of uniforms used. The optimizers `fminbnd` and `fminsearch` follow MATLAB's algorithms
and stopping rules. On the test objectives they return MATLAB's results to the last digit, and
[PARITY.md](PARITY.md) shows how closely they track MATLAB on the marginal likelihood of the
shrinkage tutorial.

The tests run on every push, together with the `your_data.py` script of every tutorial.

## Tutorials

| Question | MATLAB version |
|---|---|
| [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](tutorials/shrinkage/) | [shrinkage](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/shrinkage) |

## Citation

The MATLAB toolkit's [CITING.md](https://github.com/joshuaccchan/bvar-toolkit/blob/main/CITING.md)
names the paper behind each function, with BibTeX entries. Please cite the papers whose
methods you use.

## License

MIT; see [LICENSE](LICENSE).
