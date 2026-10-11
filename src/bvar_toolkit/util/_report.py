import datetime
import json
import os
import platform
import tempfile

import numpy as np


def _plain(x):
    """A JSON-ready copy of a settings value."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (datetime.date, datetime.datetime)):
        return x.isoformat()
    return x


def report(name, tbl, meta, outdir=None):
    """bvt.util.report - write a comparison table to <outdir>/<name>.csv and the settings to
    <outdir>/<name>.json, and print the paths. outdir defaults to the temporary directory, so a
    script that calls this leaves the working tree and a CI checkout clean.

      f = bvt.util.report(name, tbl, meta)
      f = bvt.util.report(name, tbl, meta, outdir)

      name : base name of the files
      tbl  : one DataFrame, or a dict of DataFrames when the run produces more than one shape of
             result, written one csv per entry as <name>_<key>.csv. Settings that differ by row
             belong in the columns; a csv holds its table alone, so that reading it needs no
             convention beyond pandas.read_csv.
      meta : dict describing the run (data file, sample, seeds, chain lengths, hyperparameters).
             The json file holds it with the run time, the Python, NumPy, SciPy and bvar_toolkit
             versions, and the names of the csv files.
      f    : the paths written, the csv files first and the json last
    """
    import pandas as pd
    import scipy

    from .. import __version__

    outdir = tempfile.gettempdir() if not outdir else str(outdir)
    if isinstance(tbl, pd.DataFrame):
        tables = {None: tbl}
    elif isinstance(tbl, dict) and tbl and all(isinstance(v, pd.DataFrame) for v in tbl.values()):
        tables = tbl
    else:
        raise TypeError('tbl must be a DataFrame or a nonempty dict of DataFrames')
    if not isinstance(meta, dict):
        raise TypeError('meta must be a dict')
    os.makedirs(outdir, exist_ok=True)

    files = []
    for key, t in tables.items():
        f = os.path.join(outdir, f'{name}.csv' if key is None else f'{name}_{key}.csv')
        t.to_csv(f, index=False)
        files.append(f)
    info = dict(_plain(meta))
    info['run_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    info['python_version'] = platform.python_version()
    info['numpy_version'] = np.__version__
    info['scipy_version'] = scipy.__version__
    info['bvar_toolkit_version'] = __version__
    info['tables'] = [os.path.basename(f) for f in files]
    fj = os.path.join(outdir, f'{name}.json')
    with open(fj, 'w', encoding='utf-8') as fh:
        json.dump(info, fh, indent=2)
    files.append(fj)
    print('\nreport written')
    for f in files[:-1]:
        print(f'  {f}')
    print(f'  {files[-1]}  (the settings)')
    return files
