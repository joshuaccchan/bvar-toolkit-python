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
| [How Should I Choose the Shrinkage Hyperparameters of My BVAR?](shrinkage/) | asymmetric conjugate prior, closed-form marginal likelihood, recursive forecasts |
