# Data

Each folder holds a data file of a published replication package, copied byte for byte from
the MATLAB toolkit's archive of that package (`replications/<folder>/legacy/` in
[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit)). The folder names are the
toolkit's package names.

| File | Package | Paper | md5 |
|---|---|---|---|
| `chan2019wp_acp/macrodata_Q_2018Q4.csv` | [BVAR_ACP_code.zip](https://joshuachan.org/code/BVAR_ACP_code.zip) | Chan (2019), CAMA Working Paper 51/2019 | `278bb8d073003a6f28388acc3da5cdb9` |
| `chan2022_qe_acp/database_2019Q4.xlsx` | [BVAR_ACP_R1_code.zip](https://joshuachan.org/code/BVAR_ACP_R1_code.zip) | Chan (2022), *Quantitative Economics*, 13(3): 1145-1169 | `eef96e4b91800830bdef602d3a043748` |

`macrodata_Q_2018Q4.csv` holds 238 quarters, 1959Q3 to 2018Q4, of 248 transformed series from
FRED-QD (McCracken and Ng, 2021), with no header row. `database_2019Q4.xlsx` holds the 15
quarterly series of the published version, 1985Q1 to 2019Q4, under two header rows and a date
column.
