# Parity with the MATLAB Toolkit

This file lists every function in `core/+bvar` of the MATLAB toolkit with the state of its
Python twin, and sets the numbers of each Python tutorial beside those of its MATLAB page.

## How a Function Is Checked

[`tools/matlab/export_fixtures.m`](tools/matlab/export_fixtures.m) runs the MATLAB function at
bvar-toolkit commit
[`bc98467`](https://github.com/joshuaccchan/bvar-toolkit/tree/bc98467da9f2b5bac88d5042056a066e62fdca2c)
and saves its inputs and outputs in `tests/fixtures`. The test runs the Python function on the
same inputs. Every entry of every output must agree within 1e-10 of itself, plus 1e-14 of the
largest entry, which leaves room for rounding in entries near zero. A function that draws
random numbers runs in both languages on the replay stream (`bvt.Replay` and
`tools/matlab/replay`), and it must also use as many uniforms as its MATLAB twin. The last
column gives the largest relative difference of an entry over the function's test cases.

The tests fail when a constant of a ported function moves by 1e-8, when two draws change
places, and when a sampler draws one extra uniform: twelve such changes, made to a copy of the
package on 2026-10-10, were each caught.

On a real objective an optimization can end at a slightly different point in the two
languages, because the marginal likelihood differs between MATLAB and Python in its last
digits. The forecasting exercise of the shrinkage tutorial runs both searches at 136 origins:

- `fminsearch` uses function values only to compare points, so its steps are MATLAB's unless a
  difference in the last digits reverses a comparison. The asymmetric hyperparameters equal
  MATLAB's to the last digit at 98 origins and within 1e-12 at 23 more. At the other 15 the
  search stops elsewhere within its tolerance, up to 9e-5 away in relative terms.
- `fminbnd` fits parabolas through function values, so the differences in the last digits
  enter its steps. The symmetric hyperparameter differs from MATLAB's at every origin, by at
  most 4e-8 relative.

The joint predictive likelihood of an origin then differs by at most 4e-6 relative, and every
number printed on the tutorial page is unchanged. On the test objectives of
`tests/test_optim.py`, which both languages evaluate exactly alike, both optimizers return
MATLAB's results to the last digit.

| Namespace | Function | Python | Check against MATLAB |
|---|---|---|---|
| `diag` | `geweke` | ported | same inputs; largest difference 2e-15 |
| `diag` | `inefficiency_factor` | ported | same inputs; largest difference 5e-16 |
| `diag` | `mcse` | ported | same inputs; largest difference 2e-16 |
| `diag` | `specvar0` | ported | same inputs; largest difference 4e-16 |
| `forecast` | `iterate` | not yet | |
| `forecast` | `mixquantile` | not yet | |
| `forecast` | `predictive` | ported | same inputs; largest difference 2e-16 |
| `forecast` | `realtime_loaddata` | not yet | |
| `forecast` | `simulate` | not yet | |
| `forecast` | `tables` | not yet | |
| `ml` | `acp` | ported | same inputs; largest difference 3e-16 |
| `ml` | `dirifit` | not yet | |
| `ml` | `intlike_csv` | not yet | |
| `ml` | `intlike_csv_ma` | not yet | |
| `ml` | `intlike_csv_t_ma` | not yet | |
| `ml` | `intlike_cvarsv` | not yet | |
| `ml` | `intlike_rsvar` | not yet | |
| `ml` | `intlike_t_csv` | not yet | |
| `ml` | `intlike_tvp` | not yet | |
| `ml` | `intlike_tvpsv` | not yet | |
| `ml` | `isden_arss` | not yet | |
| `ml` | `kron_bvar` | not yet | |
| `ml` | `kron_bvar_csv` | not yet | |
| `ml` | `kron_bvar_csv_ma` | not yet | |
| `ml` | `kron_bvar_csv_t_ma` | not yet | |
| `ml` | `kron_bvar_ma` | not yet | |
| `ml` | `kron_bvar_t` | not yet | |
| `ml` | `kron_bvar_t_csv` | not yet | |
| `ml` | `kron_bvar_t_ma` | not yet | |
| `ml` | `ldiripdf` | not yet | |
| `ml` | `lgampdf` | not yet | |
| `ml` | `linvgammpdf` | not yet | |
| `ml` | `llike_csv_ma` | not yet | |
| `ml` | `llike_ma` | not yet | |
| `ml` | `lmvnpdf_pcn` | not yet | |
| `ml` | `lniwpdf` | not yet | |
| `ml` | `ltnormpdf` | not yet | |
| `ml` | `mltvpsv_cvar` | not yet | |
| `ml` | `mltvpsv_cvarsv` | not yet | |
| `ml` | `mltvpsv_rs` | not yet | |
| `ml` | `mltvpsv_rs_r1` | not yet | |
| `ml` | `mltvpsv_rs_r2` | not yet | |
| `ml` | `mltvpsv_tvp` | not yet | |
| `ml` | `mltvpsv_tvp_r1_sv` | not yet | |
| `ml` | `mltvpsv_tvp_r2_sv` | not yet | |
| `ml` | `mltvpsv_tvp_r3_sv` | not yet | |
| `ml` | `mltvpsv_tvpsv` | not yet | |
| `ml` | `mlvarsv_arsv_redu` | not yet | |
| `ml` | `mlvarsv_arsvo_redu` | not yet | |
| `ml` | `mlvarsv_csv` | not yet | |
| `ml` | `mlvarsv_fsv` | not yet | |
| `models` | `mfvar_csv` | not yet | |
| `models` | `var_csv` | not yet | |
| `models` | `var_sv` | not yet | |
| `priors` | `acp_opt_kappa` | ported | same inputs; largest difference 2e-13 |
| `priors` | `acp_redu` | ported | same inputs; identical |
| `priors` | `acp_stru` | ported | same inputs; identical |
| `priors` | `impact_B0` | not yet | |
| `priors` | `minn` | not yet | |
| `priors` | `minnesota_C` | not yet | |
| `priors` | `niw` | not yet | |
| `priors` | `resid_var_allvars_ridge` | not yet | |
| `priors` | `resid_var_ar4` | ported | same inputs; largest difference 2e-16 |
| `priors` | `vtheta` | not yet | |
| `samplers` | `acp_theta_sig` | ported | replayed random numbers; largest difference 2e-13 |
| `samplers` | `alp_tri_cs` | not yet | |
| `samplers` | `eq_fsv_load` | not yet | |
| `samplers` | `eq_gauss` | not yet | |
| `samplers` | `eq_hyb_tvp` | not yet | |
| `samplers` | `eq_svar_oi` | not yet | |
| `samplers` | `eq_tri_cs` | not yet | |
| `samplers` | `eq_var_oi` | not yet | |
| `samplers` | `eq_var_redu_tri` | not yet | |
| `samplers` | `factor_fsv` | not yet | |
| `samplers` | `gig_shrinkage` | not yet | |
| `samplers` | `horseshoe_kappa_psi` | not yet | |
| `samplers` | `missing_var` | not yet | |
| `samplers` | `nu_psi_ng` | not yet | |
| `samplers` | `var_coef_joint` | not yet | |
| `structural` | `b0_row_sampler` | not yet | |
| `structural` | `check_separable` | not yet | |
| `structural` | `construct_Sigt` | not yet | |
| `structural` | `irf_redu` | not yet | |
| `structural` | `qr_sign` | not yet | |
| `structural` | `reduced_form` | ported | same inputs; largest difference 2e-14 |
| `structural` | `sign_assign` | not yet | |
| `structural` | `sign_restrict` | not yet | |
| `sv` | `csv_armh` | not yet | |
| `sv` | `csv_armh_block` | not yet | |
| `sv` | `init_approx1N` | not yet | |
| `sv` | `ksc_ar1_mean` | not yet | |
| `sv` | `ksc_rw_diffuse` | not yet | |
| `sv` | `ksc_rw_h0` | not yet | |
| `sv` | `nu_studentt` | not yet | |
| `sv` | `sv0_params` | not yet | |
| `sv` | `sv_params` | not yet | |
| `sv` | `svo_outlier` | not yet | |
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
| `util` | `surform` | ported | same inputs; identical |
| `util` | `surform2` | ported | same inputs; identical |
| `util` | `tnormrnd` | ported | replayed random numbers; largest difference 4e-16 |
| `util` | `vec` | ported | same inputs; identical |
| `util` | `vech` | ported | same inputs; identical |

## Differences from the MATLAB Functions

- Positions of variables count from 0: `idx_ns` of `priors.acp_stru`, `priors.acp_redu` and
  `priors.acp_opt_kappa`, `tgt` of `forecast.predictive`, and the segments `info.A` and
  `info.B` returned by `diag.geweke`.
- Column vectors are 1-D arrays, structs are `bvt.Struct` objects, and sparse matrices are
  `scipy.sparse.csc_array` objects.
- Name-value options are keyword arguments: `symmetric` and `ridge` of
  `priors.acp_opt_kappa`, `ridge` of `ml.acp` and `weights` of `util.mm_constraint`. The
  argument `type` of `priors.acp_opt_kappa` is called `form`.
- Every function that draws takes `rng`, in place of MATLAB's global stream.
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
  files, the legacy copies of a function in the archived packages, and the twin in
  statespace-toolkit. The docstrings of `util.gam_mode`, `util.report` and
  `structural.reduced_form` also describe their Python implementations.

## Tutorials

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
