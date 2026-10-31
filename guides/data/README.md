# Data for the Guides

[`macro3_Q.csv`](macro3_Q.csv) holds three quarterly US series from 1960Q1 to 2019Q4, 240
quarters, with each quarter dated by its first day:

| Column | Series | Source (FRED) |
|---|---|---|
| `GDPC1` | real GDP growth, annualized, in percent: 400 times the log difference of real GDP | GDPC1 |
| `PCECTPI` | PCE inflation, annualized, in percent: 400 times the log difference of the PCE chain-type price index | PCECTPI |
| `UNRATE` | the unemployment rate, in percent, averaged over the three months of the quarter | UNRATE |

The values are the first 240 rows of the GDP growth, PCE inflation and unemployment columns of
`chapter12/macro4_Q.csv` in the code repository of Chan (2027), which dates a quarter by its
last month. They agree with the FRED series as published on 15 October 2026 to within 5e-9 for
GDP growth and PCE inflation, and to the four decimals of the file for unemployment. The sample
ends before the pandemic, as in the forecasting application of that book.

The guides of the series Forecasting with a VAR fit a VAR to these data and simulate from it.

## References

Chan, J. C. C. (2027). *Bayesian Macroeconometrics: Methods and Applications*. Chapman &
Hall/CRC. [Code repository](https://github.com/joshuaccchan/bayesian-macroeconometrics)

U.S. Bureau of Economic Analysis, Real Gross Domestic Product [GDPC1] and Personal Consumption
Expenditures: Chain-type Price Index [PCECTPI]; U.S. Bureau of Labor Statistics, Unemployment Rate
[UNRATE]; retrieved from FRED, Federal Reserve Bank of St. Louis.
