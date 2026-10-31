# Parity with the MATLAB Toolkit

This file lists every function in `core/+bvar` of the MATLAB toolkit with the state of its
Python twin, sets the numbers of each Python tutorial beside those of its MATLAB page, and says
how the guides differ.

## How a Function Is Checked

[`tools/matlab/export_fixtures.m`](tools/matlab/export_fixtures.m) runs each MATLAB function at
bvar-toolkit commit [`bb03048`](https://github.com/joshuaccchan/bvar-toolkit/tree/bb03048) and
saves its inputs and outputs in `tests/fixtures`. The test runs the Python function on the same
inputs, and every entry of every output must agree within 1e-10 of itself, plus 1e-14 of the
largest entry. A function that draws random numbers runs in both languages on the same uniforms
(`bvt.Replay` and `tools/matlab/replay`) and must use as many of them as its MATLAB twin. The
last column gives the largest relative difference of an entry over the function's test cases.

Four functions end a numerical search that rounding moves slightly, and are tested at looser
tolerances: `ml.mlvarsv_arsvo_redu` at 1e-7, and the marginal likelihoods of the MA models,
`ml.kron_bvar_t_ma`, `ml.kron_bvar_csv_ma` and `ml.kron_bvar_csv_t_ma`, at 1e-6.

The tests fail when a constant moves by 1e-8, when two draws change places, when a sampler draws
one extra uniform, and when a strict inequality becomes a weak one. Of 256 such changes made to
a copy of the package, 244 were caught. Most of the 12 missed move an output by less than the
tolerance of its test.

On a real objective, an optimization can stop at a slightly different point in the two
languages, because the objective differs in its last digits. In the shrinkage tutorial, the
hyperparameters chosen at 136 origins differ from MATLAB's by at most 0.009 percent of their
value, and every number printed on the page is unchanged. In the sign-restriction tutorial, the log marginal
likelihoods at the two optima differ by 1e-5, below the tolerance of the search.

| Namespace | Function | Python | Check against MATLAB |
|---|---|---|---|
| `diag` | `geweke` | ported | same inputs; largest difference 2e-15 |
| `diag` | `inefficiency_factor` | ported | same inputs; largest difference 5e-16 |
| `diag` | `mcse` | ported | same inputs; largest difference 2e-16 |
| `diag` | `specvar0` | ported | same inputs; largest difference 4e-16 |
| `forecast` | `iterate` | ported | replayed random numbers; largest difference 3e-14 |
| `forecast` | `mixquantile` | ported | same inputs; identical |
| `forecast` | `predictive` | ported | same inputs; largest difference 2e-16 |
| `forecast` | `realtime_loaddata` | ported | same inputs; identical |
| `forecast` | `simulate` | ported | replayed random numbers; largest difference 5e-16 |
| `forecast` | `tables` | ported | same inputs; largest difference 3e-16 |
| `ml` | `acp` | ported | same inputs; largest difference 3e-16 |
| `ml` | `dirifit` | ported | same inputs; largest difference 3e-14 |
| `ml` | `intlike_csv` | ported | replayed random numbers; largest difference 5e-16 |
| `ml` | `intlike_csv_ma` | ported | replayed random numbers; largest difference 5e-16 |
| `ml` | `intlike_csv_t_ma` | ported | replayed random numbers; largest difference 6e-16 |
| `ml` | `intlike_cvarsv` | ported | replayed random numbers; largest difference 9e-16 |
| `ml` | `intlike_rsvar` | ported | same inputs; identical |
| `ml` | `intlike_t_csv` | ported | replayed random numbers; largest difference 7e-16 |
| `ml` | `intlike_tvp` | ported | same inputs; largest difference 2e-15 |
| `ml` | `intlike_tvpsv` | ported | replayed random numbers; largest difference 8e-15 |
| `ml` | `isden_arss` | ported | same inputs; largest difference 3e-13 |
| `ml` | `kron_bvar` | ported | same inputs; largest difference 2e-16 |
| `ml` | `kron_bvar_csv` | ported | replayed random numbers; largest difference 3e-14 |
| `ml` | `kron_bvar_csv_ma` | ported | replayed random numbers; largest difference 3e-08 |
| `ml` | `kron_bvar_csv_t_ma` | ported | replayed random numbers; largest difference 5e-08 |
| `ml` | `kron_bvar_ma` | ported | same inputs; largest difference 2e-14 |
| `ml` | `kron_bvar_t` | ported | same inputs; largest difference 3e-14 |
| `ml` | `kron_bvar_t_csv` | ported | replayed random numbers; largest difference 1e-14 |
| `ml` | `kron_bvar_t_ma` | ported | replayed random numbers; largest difference 5e-08 |
| `ml` | `ldiripdf` | ported | same inputs; largest difference 9e-16 |
| `ml` | `lgampdf` | ported | same inputs; largest difference 1e-13 |
| `ml` | `linvgammpdf` | ported | same inputs; identical |
| `ml` | `llike_csv_ma` | ported | same inputs; largest difference 2e-16 |
| `ml` | `llike_ma` | ported | same inputs; identical |
| `ml` | `lmvnpdf_pcn` | ported | same inputs; identical |
| `ml` | `lniwpdf` | ported | same inputs; largest difference 1e-15 |
| `ml` | `ltnormpdf` | ported | same inputs; identical |
| `ml` | `mltvpsv_cvar` | ported | replayed random numbers; largest difference 3e-15 |
| `ml` | `mltvpsv_cvarsv` | ported | replayed random numbers; largest difference 3e-15 |
| `ml` | `mltvpsv_rs` | ported | replayed random numbers; largest difference 8e-15 |
| `ml` | `mltvpsv_rs_r1` | ported | replayed random numbers; largest difference 4e-15 |
| `ml` | `mltvpsv_rs_r2` | ported | replayed random numbers; largest difference 3e-15 |
| `ml` | `mltvpsv_tvp` | ported | replayed random numbers; largest difference 7e-15 |
| `ml` | `mltvpsv_tvp_r1_sv` | ported | replayed random numbers; largest difference 5e-15 |
| `ml` | `mltvpsv_tvp_r2_sv` | ported | replayed random numbers; largest difference 3e-15 |
| `ml` | `mltvpsv_tvp_r3_sv` | ported | replayed random numbers; largest difference 2e-15 |
| `ml` | `mltvpsv_tvpsv` | ported | replayed random numbers; largest difference 5e-15 |
| `ml` | `mlvarsv_arsv_redu` | ported | replayed random numbers; largest difference 3e-14 |
| `ml` | `mlvarsv_arsvo_redu` | ported | replayed random numbers; largest difference 4e-09 |
| `ml` | `mlvarsv_csv` | ported | replayed random numbers; largest difference 1e-14 |
| `ml` | `mlvarsv_fsv` | ported | replayed random numbers; largest difference 1e-14 |
| `models` | `mfvar_csv` | ported | replayed random numbers; largest difference 2e-11 |
| `models` | `var_csv` | ported | replayed random numbers; largest difference 5e-13 |
| `models` | `var_sv` | ported | replayed random numbers; largest difference 4e-12 |
| `priors` | `acp_opt_kappa` | ported | same inputs; largest difference 2e-13 |
| `priors` | `acp_redu` | ported | same inputs; identical |
| `priors` | `acp_stru` | ported | same inputs; identical |
| `priors` | `impact_B0` | ported | same inputs; largest difference 2e-16 |
| `priors` | `minn` | ported | same inputs; largest difference 8e-14 |
| `priors` | `minnesota_C` | ported | same inputs; identical |
| `priors` | `niw` | ported | same inputs; largest difference 8e-14 |
| `priors` | `resid_var_allvars_ridge` | ported | same inputs; largest difference 7e-13 |
| `priors` | `resid_var_ar4` | ported | same inputs; largest difference 2e-16 |
| `priors` | `vtheta` | ported | same inputs; identical |
| `samplers` | `acp_theta_sig` | ported | replayed random numbers; largest difference 2e-13 |
| `samplers` | `alp_tri_cs` | ported | replayed random numbers; largest difference 1e-15 |
| `samplers` | `eq_fsv_load` | ported | replayed random numbers; largest difference 6e-14 |
| `samplers` | `eq_gauss` | ported | replayed random numbers; largest difference 2e-14 |
| `samplers` | `eq_hyb_tvp` | ported | replayed random numbers; largest difference 2e-12 |
| `samplers` | `eq_svar_oi` | ported | replayed random numbers; largest difference 3e-14 |
| `samplers` | `eq_tri_cs` | ported | replayed random numbers; largest difference 7e-14 |
| `samplers` | `eq_var_oi` | ported | replayed random numbers; largest difference 9e-15 |
| `samplers` | `eq_var_redu_tri` | ported | replayed random numbers; largest difference 7e-15 |
| `samplers` | `factor_fsv` | ported | replayed random numbers; largest difference 4e-14 |
| `samplers` | `gig_shrinkage` | ported | replayed random numbers; identical |
| `samplers` | `horseshoe_kappa_psi` | ported | replayed random numbers; largest difference 1e-15 |
| `samplers` | `missing_var` | ported | replayed random numbers; largest difference 4e-13 |
| `samplers` | `nu_psi_ng` | ported | replayed random numbers; largest difference 3e-15 |
| `samplers` | `var_coef_joint` | ported | replayed random numbers; largest difference 6e-14 |
| `structural` | `b0_row_sampler` | ported | replayed random numbers; largest difference 2e-14 |
| `structural` | `check_separable` | ported | same inputs; identical |
| `structural` | `construct_Sigt` | ported | same inputs; largest difference 7e-16 |
| `structural` | `irf_redu` | ported | same inputs; largest difference 2e-15 |
| `structural` | `qr_sign` | ported | same inputs; largest difference 4e-14 |
| `structural` | `reduced_form` | ported | same inputs; largest difference 2e-14 |
| `structural` | `sign_assign` | ported | replayed random numbers; identical |
| `structural` | `sign_restrict` | ported | same inputs; identical |
| `sv` | `csv_armh` | ported | replayed random numbers; largest difference 8e-14 |
| `sv` | `csv_armh_block` | ported | replayed random numbers; largest difference 3e-14 |
| `sv` | `init_approx1N` | ported | same inputs; largest difference 9e-13 |
| `sv` | `ksc_ar1_mean` | ported | replayed random numbers; largest difference 7e-14 |
| `sv` | `ksc_rw_diffuse` | ported | replayed random numbers; largest difference 8e-14 |
| `sv` | `ksc_rw_h0` | ported | replayed random numbers; largest difference 2e-13 |
| `sv` | `nu_studentt` | ported | replayed random numbers; largest difference 1e-14 |
| `sv` | `sv0_params` | ported | replayed random numbers; largest difference 1e-15 |
| `sv` | `sv_params` | ported | replayed random numbers; largest difference 7e-16 |
| `sv` | `svo_outlier` | ported | replayed random numbers; largest difference 2e-14 |
| `util` | `anormrnd` | ported | replayed random numbers; largest difference 4e-16 |
| `util` | `build_lags` | ported | same inputs; identical |
| `util` | `diffmat` | ported | same inputs; identical |
| `util` | `dirirnd` | ported | replayed random numbers; largest difference 1e-15 |
| `util` | `dlog_gaps` | ported | same inputs; identical |
| `util` | `gam_mode` | ported | same inputs; identical |
| `util` | `igrnd` | ported | replayed random numbers; largest difference 6e-16 |
| `util` | `kron_gram` | ported | same inputs; largest difference 3e-15 |
| `util` | `ldet` | ported | same inputs; largest difference 3e-16 |
| `util` | `logsumexp` | ported | same inputs; identical |
| `util` | `mgammaln` | ported | same inputs; largest difference 3e-16 |
| `util` | `mm_constraint` | ported | same inputs; identical |
| `util` | `report` | ported | writes csv files and a json file of settings; tested in Python (`test_report`) |
| `util` | `select_obs` | ported | same inputs; identical |
| `util` | `shaded_band` | ported | matplotlib drawing; tested in Python (`test_shaded_band`) |
| `util` | `simulate_var` | ported | replayed random numbers; largest difference 2e-15 |
| `util` | `surform` | ported | same inputs; identical |
| `util` | `surform2` | ported | same inputs; identical |
| `util` | `tnormrnd` | ported | replayed random numbers; largest difference 4e-16 |
| `util` | `vec` | ported | same inputs; identical |
| `util` | `vech` | ported | same inputs; identical |

## Differences from the MATLAB Functions

- Positions count from 0: `idx_ns` of `priors.acp_stru`, `priors.acp_redu` and
  `priors.acp_opt_kappa`, `tgt` of `forecast.predictive`, `Ridx` of `structural.sign_restrict`,
  the segments `info.A` and `info.B` returned by `diag.geweke`, the pairs returned by
  `structural.check_separable`, the index sets `idx_kappa1` and `idx_kappa2` of
  `priors.minnesota_C`, `priors.vtheta`, `samplers.horseshoe_kappa_psi` and
  `samplers.gig_shrinkage`, `var_core` of `forecast.tables`, and the mixture indicators
  returned by `sv.ksc_ar1_mean` and `sv.ksc_rw_diffuse`, which run from 0 to 6.
- The periods `t` and `T` of `forecast.iterate` and `t` and `T0` of `forecast.realtime_loaddata`
  count from 1, as the periods of the real-time data files do, so `forecast.iterate` reads the
  realized values of period `t + 1` from row `t` of `Y`.
- When two shocks draw the same column, `structural.sign_assign` raises a ValueError that
  names the cause; the MATLAB function stops on a size mismatch.
- Column vectors are 1-D arrays, structs are `bvt.Struct` objects, and sparse matrices are
  `scipy.sparse.csc_array` objects.
- Name-value options are keyword arguments with MATLAB's names, such as `symmetric` and
  `ridge` of `priors.acp_opt_kappa`, `M` and `z` of `samplers.missing_var` and `draws` of the
  models. Four are renamed to Python's conventions: `O` of `samplers.missing_var` is `o`;
  `ForcedAccept`, `MaxIterMode` and `MaxIterAR` of `sv.csv_armh_block` and `sv.csv_armh` are
  `forced_accept`, `max_iter_mode` and `max_iter_ar`, the first two of them positional in
  `sv.csv_armh` as in MATLAB. The argument `type` of `priors.acp_opt_kappa` is called `form`.
- Every function that draws takes `rng`, in place of MATLAB's global stream. In the models it
  replaces the `'seed'` option, and `res.seed` records `rng` when it is an integer.
- `samplers.var_coef_joint` always returns the posterior mean as its second output; MATLAB
  computes it only when asked for it.
- `samplers.missing_var` returns the draws as a 1-D array when `ndraws` is 1, and
  `sv.sv0_params` returns scalars when `h` is one series given as a 1-D array.
- `models.mfvar_csv` keeps the draws of the missing values in float32, as MATLAB keeps them in
  single precision, and computes their quantiles with `method='hazen'`, which is MATLAB's.
- The precision samplers factor their banded matrices with LAPACK's banded Cholesky
  factorization (`bvar_toolkit._linalg`), which gives the factor of MATLAB's sparse `chol` to
  rounding, and solve with many right-hand sides by blocks of rows.
- `structural.b0_row_sampler` computes null spaces as MATLAB's `null` does, from the singular
  value decomposition of LAPACK's `gesvd`, whose singular vectors have MATLAB's signs. The
  signs enter the draw, and NumPy's default `gesdd` gives other signs for some matrices.
- The marginal likelihoods of `ml` fit gamma and beta distributions with the algorithms of
  MATLAB's `gamfit` and `betafit` (`bvar_toolkit._stats`): `fzero` to a tolerance of 1e-8 and
  `fminsearch` to 1e-6, both reimplemented in `bvar_toolkit._optim`. The fits add up their
  data in MATLAB's order and evaluate MATLAB's `gammaln`, which differs from SciPy's in the
  last digit at about half of the points tested; with NumPy's sums and SciPy's `gammaln`, a
  beta fit ends 8e-7 from MATLAB's.
- An inverse-Wishart draw (`Stream.iwishrnd`) uses the Bartlett decomposition, in place of the
  algorithm of MATLAB's `iwishrnd`; the replay versions in both languages use the Bartlett
  decomposition.
- A generalized inverse Gaussian draw (`Stream.gigrnd`) comes from SciPy's `geninvgauss`, in place
  of the algorithm of `third_party/gigrnd.m`, which `samplers.gig_shrinkage` calls in MATLAB; the
  replay versions in both languages return the mode of the distribution.
- The marginal likelihoods of the MA models of Chan (2020) call MATLAB's `fminunc`, whose line
  search is not published. In its place, `bvar_toolkit._optim.fminunc` finds the minimum by
  `fminsearch` at tolerances of 1e-10 and returns the finite-difference second derivative of
  `fminunc`, by its formula. On the test objectives its minimizer lies within 2e-8 of
  `fminunc`'s and nearer the exact one, and the second derivatives agree to 3e-9. For the tests,
  MATLAB runs the same search (`tools/matlab/replay/fminunc.m`).
- The grids of the marginal likelihoods come from `bvar_toolkit._stats.matlab_linspace`, which
  computes them with the arithmetic of MATLAB's `linspace` and equals it digit for digit.
- `priors.acp_opt_kappa` searches with `bvar_toolkit._optim`, which implements the algorithms,
  defaults and stopping rules of MATLAB's `fminbnd` and `fminsearch`.
- `util.logsumexp` takes `axis`, counted from 0, in place of `dim`, and treats a 1-D array as
  one column.
- `util.select_obs` always returns `yo` as its third output.
- `util.ldet` factors a sparse matrix as a dense one.
- `util.report` writes the settings to a json file in place of a mat file, with the Python,
  NumPy and SciPy versions in place of the MATLAB version and the last warning.
- `util.shaded_band` draws on the axes `ax` when given one.
- Each docstring is the header of its MATLAB twin, with Python's call syntax, indexing and
  `rng`. Notes that concern MATLAB alone are left out: toolbox requirements, MATLAB test
  files and measurements, the replication folders, the legacy copies of a function in the
  archived packages, the twin in statespace-toolkit, and the shapes of MATLAB arrays. The
  samplers that update their array inputs say that they leave them unmodified, and the
  docstring of `util.report` describes its Python implementation.

## Tutorials

### Does the Order of the Variables Change My VAR Results?

The MATLAB column is the [MATLAB page](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/variable_ordering);
the Python column is the output of `tutorials/variable_ordering/build.py`. The 20-variable results
and the forecast comparison come from the files of the replication package, so they are the same
on both pages, and the build checks the forecast comparison against the package's own run,
number for number. The four-variable chains come from MATLAB's and NumPy's own generators, so
Table 1 agrees within Monte Carlo error.

| Result | MATLAB | Python |
|---|---|---|
| Table 1, Cholesky, order reversed: the four variances | 4.1%, 7.7%, 0.9%, 1.7% | 4.3%, 7.7%, 1.0%, 1.6% |
| Table 1, Cholesky, order reversed: the six correlations | 0.111, 0.067, 0.088, 0.004, 0.086, 0.022 | 0.111, 0.068, 0.088, 0.004, 0.086, 0.022 |
| Table 1, Cholesky, seed changed: largest variance, largest correlation | 0.8%, 0.001 | 1.1%, 0.001 |
| Table 1, order-invariant, order reversed: largest variance, largest correlation | 1.1%, 0.001 | 1.2%, 0.001 |
| Table 1, order-invariant, seed changed: largest variance, largest correlation | 1.2%, 0.001 | 1.0%, 0.001 |
| Tables 2 to 4 | | the same at every printed digit |
| Build time | 22.5 minutes, MATLAB R2025b | 8.4 minutes, Python 3.13 |

### How Should I Choose the Shrinkage Hyperparameters of My BVAR?

The MATLAB column is the [MATLAB page](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/shrinkage);
the Python column is the output of `tutorials/shrinkage/build.py`. The first block involves
no random draws. The second block depends on posterior draws, which come from MATLAB's and
NumPy's own generators, so its numbers agree within Monte Carlo error.

| Result | MATLAB | Python |
|---|---|---|
| Table 1: symmetric $`\kappa`$, log marginal likelihood | 0.039, -9436.3 | 0.039, -9436.3 |
| Table 1: subjective prior, log marginal likelihood | -9371.9 | -9371.9 |
| Table 1: asymmetric $`(\kappa_2, \kappa_3)`$, log marginal likelihood | (0.406, 0.0092), -9201.1 | (0.406, 0.0092), -9201.1 |
| 90% grid intervals of $`\kappa_2`$ and $`\kappa_3`$ | [0.32, 0.56], [0.0074, 0.0114] | [0.32, 0.56], [0.0074, 0.0114] |
| 15-variable application: asymmetric minus symmetric | 8.3 | 8.3 |
| Table 2: eight lag lengths | | the same at every printed digit |
| Optimal $`\kappa_2`$ and $`\kappa_3`$ over the 136 origins | 0.405–0.487, 0.0086–0.0135 | 0.405–0.487, 0.0086–0.0135 |
| Joint log predictive likelihoods: symmetric, subjective, asymmetric | -5112.4, -5062.9, -4974.1 | -5112.4, -5062.9, -4974.1 |
| Diebold-Mariano, joint: against symmetric, against subjective | 7.1, 6.3 | 7.10, 6.32 |
| Other-lag coefficients, asymmetric against symmetric | 34% smaller | 34% smaller |
| Own-lag coefficients, asymmetric against symmetric | 31% larger | 31% larger |
| Table 3, symmetric, h = 1: median RMSFE gain; gains; significant gains / losses | 3.61; 20; 8 / 0 | 3.59; 20; 8 / 0 |
| Table 3, symmetric, h = 1: median ALPL gain; gains; significant gains / losses | 2.55; 18; 5 / 0 | 2.50; 18; 5 / 0 |
| Table 3, symmetric, h = 4: RMSFE | 0.52; 11; 2 / 0 | 0.52; 12; 2 / 0 |
| Table 3, symmetric, h = 4: ALPL | 1.17; 13; 4 / 0 | 1.18; 13; 4 / 0 |
| Table 3, subjective, h = 1: RMSFE | 1.93; 14; 0 / 0 | 1.98; 14; 0 / 0 |
| Table 3, subjective, h = 1: ALPL | 3.55; 15; 11 / 0 | 3.56; 17; 11 / 0 |
| Table 3, subjective, h = 4: RMSFE | -0.36; 10; 0 / 2 | -0.34; 10; 0 / 2 |
| Table 3, subjective, h = 4: ALPL | 1.11; 12; 1 / 0 | 1.11; 13; 1 / 0 |
| Build time | 12.7 minutes, MATLAB R2025b | 10.1 minutes, Python 3.13 |

### Which Stochastic Volatility Specification Should My VAR Use?

The MATLAB column is the [MATLAB page](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/sv_specification)
and its saved runs; the Python column is the output of `tutorials/sv_specification/build.py`.
VAR-NCP's marginal likelihood is in closed form, and the two agree. The chains and the importance
samples come from MATLAB's and NumPy's own generators, and the shrinkage hyperparameters are drawn
by SciPy's `geninvgauss` in place of the package's `gigrnd`, so the other numbers agree within
Monte Carlo error. `tests/test_drivers.py` runs `ml_varsv.py` beside the package's `run_ml` on the
replay stream, with the generalized inverse Gaussian draws replaced by the mode of their
distribution in both languages, and the seven specifications agree draw for draw. In the build the
outlier probability of 2020Q1 is 0.56 and 0.53 in the two runs against 0.49 and 0.49 in MATLAB's,
which puts it above one half.

| Result | MATLAB | Python |
|---|---|---|
| Table 1: VAR-NCP | -1587.4 | -1587.4 |
| Table 1: VAR-CSV, VAR-SV, VAR-FSV with one and two factors, VAR-SVO | -1129.7, -1018.7, -1022.0, -1026.8, -1008.8 | -1129.6, -1019.1, -1022.4, -1027.1, -1008.1 |
| Numerical standard errors | 0.08 to 0.51 | 0.12 to 0.38 |
| VAR-SV under the symmetric prior; what the restriction costs | -1084.3; 65.7 | -1084.7; 65.7 |
| Own-lag and other-lag hyperparameters of VAR-SV | 0.244, 0.00090 | 0.244, 0.000868 |
| Quarters with an outlier probability above one half | 2008Q4, 2020Q2, 2020Q3 | 2008Q4, 2020Q1, 2020Q2, 2020Q3 |
| Outlier probability of 2020Q1 in the two runs | 0.49, 0.49 | 0.56, 0.53 |
| Posterior mean of $`p_o`$; expected number of outlier quarters | 0.026; 4.0 | 0.0265; 4.1 |
| VAR-CSV multiplier, lowest and highest | 0.36 in 2018Q2, 10.64 in 2020Q2 | 0.36 in 2018Q2, 10.81 in 2020Q2 |
| Largest difference between the two seeds | 0.7 | 0.4 |
| VAR-FSV with one factor and GDP growth first, two seeds | -1053.4, -1044.3 | -1056.1, -1027.6 |
| Largest gap between the posterior means of the two runs | 0.13 | 0.28 |
| Build time | 12.8 minutes, MATLAB R2025b | 19.1 minutes, Python 3.13 |

### Does Modeling the Volatility Improve My Forecasts?

The MATLAB column is the [MATLAB page](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/forecasting);
the Python column is the output of `tutorials/forecasting/build.py`, `forecast_now.py`,
`bench_density.py` and `check_stability.py`. The posterior draws come from MATLAB's and NumPy's
own generators, so the two agree within Monte Carlo error. At one quarter the median absolute
difference between the two builds in the joint log score of a forecast is 0.005 for the
homoskedastic model, 0.02 for VAR-CSV and 0.06 for VAR-OISV. The largest differences are at the
2020 targets, whose realized values lie far in the tails of every predictive density, so that the
average over the draws is set by a few of them.

| Result | MATLAB | Python |
|---|---|---|
| Table 1, VAR-CSV, log score gain, h = 1: all; through 2019; from 2020 | 5.76; 1.63; 30.50 | 5.39; 1.59; 28.16 |
| Table 1, VAR-CSV, log score gain, h = 4 | 2.68; 1.43; 9.98 | 2.52; 1.43; 8.91 |
| Table 1, VAR-OISV, log score gain, h = 1 | 5.62; 2.21; 26.11 | 5.52; 2.20; 25.42 |
| Table 1, VAR-OISV, log score gain, h = 4 | 2.07; 1.61; 4.81 | 1.90; 1.60; 3.66 |
| Table 1, RMSFE gain over all forecasts, h = 1 and 4: VAR-CSV; VAR-OISV | 4.9%, 5.0%; 8.5%, 17.6% | 5.0%, 4.9%; 8.7%, 17.6% |
| Joint log score of the 2020Q2 target, h = 1: homoskedastic, VAR-CSV, VAR-OISV | −624.9, −56.1, −170.1 | −616.2, −90.3, −171.4 |
| Median predictive sd of GDP growth through 2019: homoskedastic, VAR-CSV, VAR-OISV | 2.80, 2.05, 2.36 | 2.80, 2.04, 2.37 |
| 80% coverage for GDP growth, h = 1: homoskedastic, VAR-CSV, VAR-OISV | 87%, 76%, 83% | 87%, 77%, 83% |
| Table 5, probability of negative GDP growth in 2025Q1 | 32%, 5%, 9% | 32%, 5%, 9% |
| Pandemic origin, h = 4: path-based and conditional Gaussian score; seed ranges | −19.83, −18.28; 1.30, 0.32 | −18.91, −18.39; 8.88, 0.35 |
| Inefficiency factors at the last origin: VAR-CSV; VAR-OISV | 8, 23, 62; 21, 62, 12 | 11, 19, 76; 22, 12, 24 |
| Stability check, largest seed range at 2005Q4, 2015Q4, 2019Q4 and 2020Q2: h = 1; h = 4 | 0.23; 0.55 | 0.55; 0.69 |
| Build time | 23.0 minutes, MATLAB R2025b | 41.4 minutes, Python 3.13 |

### Can I Use Many Sign Restrictions in My Large VAR?

The MATLAB column is the [MATLAB page](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/sign_restrictions)
and its saved bands; the Python column is the output of `tutorials/sign_restrictions/build.py`.
The candidate rotations come from MATLAB's and NumPy's own generators, so the acceptance counts
and the impulse responses agree within Monte Carlo error.

| Result | MATLAB | Python |
|---|---|---|
| Sign and ranking restrictions, checked against the package's driver | 98, 7 | 98, 7 |
| Pairs of shocks separated by their signs, only by the ranking restrictions | 23, 5 | 23, 5 |
| Shrinkage on own lags, on cross lags; log marginal likelihood | 0.2444, 0.0025; -6613.0 | 0.2444, 0.002499; -6613.0 |
| Candidates for 100 admissible draws; candidates per draw | 776,000; 7,760 | 872,000; 8,720 |
| Candidates accepted by the rule of Rubio-Ramírez, Waggoner and Zha | 0 | 0 |
| Table 1: largest difference between the two medians | | 0.058, investment under the monetary shock |
| MATLAB medians inside the Python 68% bands: Table 1; all 35 x 8 x 36 responses | | 48 of 48; all |
| Impact gaps between medians: financial, demand, government spending | 0.160, −0.084, 0.043 | 0.177, −0.075, 0.039 |
| Build time | 9.8 minutes, MATLAB R2025b | 8.1 minutes, Python 3.13 |

### How Do I Handle Missing and Mixed-Frequency Data in My VAR?

The MATLAB column is the [MATLAB page](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials/mixed_frequency);
the Python column is the output of `tutorials/mixed_frequency/build.py`. The posterior draws come
from MATLAB's and NumPy's own generators, so the two agree within Monte Carlo error.

| Result | MATLAB | Python |
|---|---|---|
| Monthly GDP growth, April 2020, with its 90% band | −7.92 (−13.3, −2.8) | −8.01 (−13.5, −2.7) |
| Monthly GDP growth, June 2020 | 4.22 | 4.38 |
| Monthly investment growth, April 2020, with its 90% band | −10.34 (−21.5, 0.3) | −10.18 (−21.4, 0.6) |
| Correlation with the Brave-Butters-Kelley series; without 2020 | 0.93; 0.91 | 0.92; 0.91 |
| Root mean squared difference from it; without 2020 | 0.19; 0.14 | 0.20; 0.14 |
| Unemployment rate, October 2025 | 4.43 | 4.43 |
| CPI inflation, October and November 2025 | 0.136, 0.116 | 0.137, 0.115 |
| Proposals accepted: blocks of 36 months; the whole path in one step | 95.5%; 0.7% | 95.5%; 1.4% |
| Largest violation of the aggregation constraints | 1.0e-13 | 7.9e-14 |
| Inefficiency factors, median and largest: GDP, investment, volatility | 5.4, 22.5; 6.0, 20.4; 12.5, 25.0 | 5.5, 21.8; 5.6, 19.5; 15.6, 30.4 |
| Build time | 24.6 minutes, MATLAB R2025b | 26.3 minutes, Python 3.13 |

## Guides

The guides generate their data with NumPy's random number generator, so a seed gives a
different sample from MATLAB's, and the numbers on each page come from the Python sample.
Where that sample shows a different pattern, the page says what it shows, so some sentences
of `coefficient_draw`, `gibbs_sampler`, `var_lag_shrinkage`, `var_forecast` and
`var_forecast_evaluation` differ from their MATLAB pages. Numbers computed from the data in
`guides/data` alone, such as the VAR fitted in `var_simulation`, equal MATLAB's to the printed
digits.
