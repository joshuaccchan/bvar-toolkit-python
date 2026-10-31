"""bvar-toolkit for Python: large Bayesian VARs.

The functions are those of the MATLAB toolkit (https://github.com/joshuaccchan/bvar-toolkit),
namespace for namespace: bvar.util.build_lags in MATLAB is bvar_toolkit.util.build_lags here.
Positions of variables count from 0, and every function that draws random numbers takes an
rng argument (None, an integer seed, a numpy.random.Generator or a Stream).
"""

__version__ = '0.1.0.dev0'

from ._rng import Replay, Stream
from ._struct import Struct
from . import diag, forecast, ml, models, priors, samplers, structural, sv, util

__all__ = ['Replay', 'Stream', 'Struct', 'diag', 'forecast', 'ml', 'models', 'priors',
           'samplers', 'structural', 'sv', 'util']
