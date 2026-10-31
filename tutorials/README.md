# Tutorials

Each tutorial answers one empirical question with the settings of a published paper, on data
in this repository. A tutorial is a page, `README.md`; a script, `build.py`, that computes
every number and figure on the page; and a script, `your_data.py`, that runs the same analysis
on any csv file, with its settings at the top. Run them from the root of the repository.

These are the Python versions of the tutorials of
[bvar-toolkit](https://github.com/joshuaccchan/bvar-toolkit/tree/main/tutorials). Each page
reports the numbers of its own build. Results that involve random draws differ from those on
the MATLAB page within Monte Carlo error, and the others agree at the printed precision.

| Tutorial | Model and method |
|---|---|
| [Does the Order of the Variables Change My VAR Results?](variable_ordering/) | Cholesky and order-invariant stochastic volatility, the effect of the ordering on estimates and forecasts |
| [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](shrinkage/) | asymmetric conjugate prior, closed-form marginal likelihood, recursive forecasts |
| [Which Stochastic Volatility Specification Should My VAR Use?](sv_specification/) | common, Cholesky, factor and outlier stochastic volatility, marginal likelihoods by importance sampling |
| [Does Modeling the Volatility Improve My Forecasts?](forecasting/) | recursive forecasts from a homoskedastic VAR and two volatility models, log predictive likelihood, calibration |
| [Can I Use Many Sign Restrictions in My Large VAR?](sign_restrictions/) | sign and ranking restrictions, the assignment of shocks to columns, impulse responses |
| [How Do I Handle Missing and Mixed-Frequency Data in My VAR?](mixed_frequency/) | mixed-frequency VAR with common stochastic volatility, missing values drawn in one block |
