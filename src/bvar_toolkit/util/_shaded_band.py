import numpy as np


def shaded_band(x, lo, hi, shade=0.85, ax=None):
    """bvt.util.shaded_band - a pointwise credible band drawn as a shaded region.

      h = bvt.util.shaded_band(x, lo, hi)
      h = bvt.util.shaded_band(x, lo, hi, shade)
      h = bvt.util.shaded_band(x, lo, hi, shade, ax)

      x      : points on the horizontal axis, numeric or dates
      lo, hi : lower and upper bounds at those points, the same length as x
      shade  : grey level in [0, 1] (default .85)
      ax     : matplotlib axes to draw on (default the current axes)
      h      : the PolyCollection drawn

    Plot lines after the band to draw them over it. The function is shaded_band.m from the code
    of Bayesian Macroeconometrics: Methods and Applications (chapter 14), in Python.
    """
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax
    return ax.fill_between(np.ravel(x), np.ravel(lo), np.ravel(hi), facecolor=(shade,) * 3,
                           edgecolor='none', linewidth=0)
