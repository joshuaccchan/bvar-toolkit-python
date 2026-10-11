"""Random draws under MATLAB's names and parameterizations.

Every function that draws takes an argument rng: None for a fresh generator, an integer
seed, a numpy.random.Generator, or a Stream. Sizes follow NumPy, so randn(n) is a vector
of n draws, and gamrnd takes MATLAB's shape and scale.

Replay(seed) draws from the uniforms of MATLAB's RandStream('mt19937ar', 'Seed', seed),
which NumPy's RandomState(seed) reproduces bit for bit, and turns them into normal, gamma
and beta draws by their inverse distribution functions, filling arrays in MATLAB's
column-major order. The functions in tools/matlab/replay do the same in MATLAB, so a
function and its MATLAB twin can be run on the same random numbers. Replay serves the
tests; it is slower than NumPy's own samplers.
"""

import numpy as np
from scipy import special


def _size(size):
    """NumPy size from the arguments of rand(*size): None for one draw."""
    if len(size) == 0:
        return None
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(int(s) for s in size[0])
    return tuple(int(s) for s in size)


class Stream:
    """Draws from a numpy.random.Generator under MATLAB's function names."""

    def __init__(self, gen=None):
        self.gen = np.random.default_rng() if gen is None else gen

    def rand(self, *size):
        """Uniform draws on (0, 1)."""
        return self.gen.random(_size(size))

    def randn(self, *size):
        """Standard normal draws."""
        return self.gen.standard_normal(_size(size))

    def gamrnd(self, a, b, *size):
        """Gamma draws with shape a and scale b; without a size, one per element of a and b."""
        return self.gen.gamma(a, b, _size(size))

    def betarnd(self, a, b, *size):
        """Beta draws; without a size, one per element of a and b."""
        return self.gen.beta(a, b, _size(size))


class Replay(Stream):
    """Draws from the MATLAB-compatible uniform stream of a seed; count is the number of
    uniforms used so far."""

    def __init__(self, seed):
        self._rs = np.random.RandomState(int(seed))
        self.count = 0

    def _uniforms(self, shape):
        k = int(np.prod(shape)) if shape else 1
        self.count += k
        u = self._rs.random_sample(k)
        if not shape:
            return u[0]
        return u.reshape(shape, order='F')

    def rand(self, *size):
        return self._uniforms(_size(size))

    def randn(self, *size):
        return special.ndtri(self._uniforms(_size(size)))

    def gamrnd(self, a, b, *size):
        shape = _size(size)
        if shape is None:
            shape = np.broadcast_shapes(np.shape(a), np.shape(b))
        return special.gammaincinv(a, self._uniforms(shape)) * b

    def betarnd(self, a, b, *size):
        shape = _size(size)
        if shape is None:
            shape = np.broadcast_shapes(np.shape(a), np.shape(b))
        return special.betaincinv(a, b, self._uniforms(shape))


def as_rng(rng=None):
    """The Stream behind an rng argument."""
    if isinstance(rng, Stream):
        return rng
    if isinstance(rng, np.random.Generator):
        return Stream(rng)
    if rng is None or isinstance(rng, (int, np.integer)):
        return Stream(np.random.default_rng(rng))
    raise TypeError('rng must be None, an integer seed, a numpy.random.Generator or a Stream')
