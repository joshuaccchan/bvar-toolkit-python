# Data

Each folder holds a data file of a published replication package, copied byte for byte from
the MATLAB toolkit's archive of that package (`replications/<folder>/legacy/` in
[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit)). The folder names are the
toolkit's package names.

| File | Package | Paper | md5 |
|---|---|---|---|
| `chan2019wp_acp/macrodata_Q_2018Q4.csv` | [BVAR_ACP_code.zip](https://joshuachan.org/code/BVAR_ACP_code.zip) | Chan (2019), CAMA Working Paper 51/2019 | `278bb8d073003a6f28388acc3da5cdb9` |
| `chan2022_qe_acp/database_2019Q4.xlsx` | [BVAR_ACP_R1_code.zip](https://joshuachan.org/code/BVAR_ACP_R1_code.zip) | Chan (2022), *Quantitative Economics*, 13(3): 1145-1169 | `eef96e4b91800830bdef602d3a043748` |
| `chan_matthes_yu2026_qe_svarsign/data35_1975.csv` | [ReplicationPackage (2026Aug).zip](https://joshuachan.org/code/ReplicationPackage%20(2026Aug).zip) | Chan, Matthes and Yu (2026), *Quantitative Economics*, 17(3): 709-740 | `72c528b372f98e2b672b2b7fa2990767` |
| `chan_matthes_yu2026_qe_svarsign/main_35VAR_Figs2to10_except5.m` | the same | the same | `875ba33233c3d305ac09269cb7fff84d` |
| `chan_koop_yu2024_jbes_oisv/FRED_MD_20vars.csv` | [OISV_code.zip](https://joshuachan.org/code/OISV_code.zip) | Chan, Koop and Yu (2024), *Journal of Business and Economic Statistics*, 42(2): 825-837 | `8e3e1900caa21309ec5c3422f86c5849` |
| `chan_koop_yu2024_jbes_oisv/golden_log_Table3_forecasting.txt` | the run of the package's `Table3_forecasting.m` captured in bvar-toolkit | the same | `3a0ce9967bebf740d00d6defff6a430e` |

`macrodata_Q_2018Q4.csv` holds 238 quarters, 1959Q3 to 2018Q4, of 248 transformed series from
FRED-QD (McCracken and Ng, 2021), with no header row. `database_2019Q4.xlsx` holds the 15
quarterly series of the published version, 1985Q1 to 2019Q4, under two header rows and a date
column. `data35_1975.csv` holds 35 quarterly US series, 1975Q1 to 2020Q4, under a row of names
and a row of transformations; the sign-restriction tutorial uses the block from 1983Q1 to
2019Q4. `main_35VAR_Figs2to10_except5.m` is the package's MATLAB driver; that tutorial reads
the sign and ranking restrictions it states and checks its own against them.

`FRED_MD_20vars.csv` holds 730 months, 1959:03 to 2019:12, of the 20 transformed FRED-MD series
of Chan, Koop and Yu (2024), with no header row; the variable-ordering tutorial uses the columns
counted 3, 5, 11 and 12 from 0. `golden_log_Table3_forecasting.txt` is the output of the
package's `Table3_forecasting.m`, run in MATLAB R2025b on the forecasts the package ships and
captured in `tests/golden` of bvar-toolkit; that tutorial checks its own recomputation of the
table against it. The package's `results_mat` folder, eight files and 32 MB, is not copied here:
the tutorial downloads it on first use into `chan_koop_yu2024_jbes_oisv/results_mat/` from the
archive of the package in bvar-toolkit, at the tag `as-published/chan_koop_yu2024_jbes_oisv`, and
checks each file against its md5 sum.
