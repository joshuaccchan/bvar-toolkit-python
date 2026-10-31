"""Maximum likelihood fits of MATLAB's Statistics and Machine Learning Toolbox, with its
algorithms, so that a fit here equals its MATLAB twin to rounding."""

import warnings
from decimal import Decimal, getcontext

import numpy as np
from scipy.special import digamma

from ._optim import fminsearch, fzero


def matlab_sum(v):
    """The sum of a vector in MATLAB's order: four running sums over the elements in strides of
    four, added together in order, then the elements left over. The fits start their searches
    from sums, so that a difference in the last digit of a sum can end a search elsewhere
    within its tolerance. The order is that of MATLAB R2025b on an x86-64 processor; MATLAB
    may sum in another order on other platforms."""
    v = np.asarray(v, dtype=float).ravel()
    m = v.size - v.size % 4
    s = 0.0
    if m:
        for lane in np.cumsum(v[:m].reshape(-1, 4), axis=0)[-1]:
            s += lane
    for t in v[m:]:
        s += t
    return s


def matlab_linspace(d1, d2, n):
    """n points from d1 to d2 with the arithmetic of MATLAB's linspace, which computes
    k*(d2/(n-1)) on an interval symmetric about zero and d1 + k*(d2 - d1)/(n-1) otherwise, and
    sets the end points, and the middle point of a symmetric grid, exactly."""
    n = int(n)
    n1 = n - 1
    if d1 == -d2 and n > 2:
        y = np.arange(-n1, n1 + 1, 2) * (d2 / n1)
        y[0], y[-1] = d1, d2
        if n1 % 2 == 0:
            y[n1 // 2] = 0.0
        return y
    y = d1 + np.arange(n1 + 1) * (d2 - d1) / n1
    if d1 == d2:
        y[:] = d1
    elif y.size:
        y[0], y[-1] = d1, d2
    return y


def gamfit(x):
    """The maximum likelihood estimate (shape, scale) of a gamma distribution from the positive
    data x, as MATLAB's gamfit for uncensored data: the data are scaled to mean one, the shape
    solves log(a) - psi(a) = log(mean(x)) - mean(log(x)) by fzero to a tolerance of 1e-8 from
    a bracket around the method-of-moments estimate, and the scale is mean(x)/a."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0 or not np.all(np.isfinite(x)):
        return np.array([np.nan, np.nan])
    if np.any(x < 0):
        raise ValueError('gamfit needs nonnegative data')
    scalex = matlab_sum(x) / n
    x = x / scalex
    s2 = matlab_sum((x - 1.0)**2) / n
    if s2 <= 100 * np.spacing(1.0):                 # constant data
        return np.array([np.inf, 0.0])
    s2 = s2 * n / (n - 1)
    ahat = 1.0 / s2                                 # the method-of-moments estimates
    if np.any(x == 0):
        warnings.warn('the data contain zeros; returning the method-of-moments estimates')
        return np.array([ahat, s2 * scalex])
    const = matlab_sum(np.log(x)) / n - np.log(n / n)

    def lkeqn(a):
        return -const - np.log(a) + digamma(a)

    if lkeqn(ahat) > 0:
        upper = ahat
        lower = .5 * upper
        while lkeqn(lower) > 0:
            upper = lower
            lower = .5 * upper
    else:
        lower = ahat
        upper = 2 * lower
        while lkeqn(upper) < 0:
            lower = upper
            upper = 2 * lower
    ahat = fzero(lkeqn, (lower, upper), tolx=1e-8)[0]
    return np.array([ahat, 1.0 / ahat * scalex])


def betaln(a, b):
    """log B(a, b) as MATLAB computes it, gammaln(a) + gammaln(b) - gammaln(a + b) with MATLAB's
    gammaln. The terms cancel, so a difference of one unit in the last place of gammaln, which
    SciPy's has against MATLAB's near 800, grows by the size of the terms."""
    return matlab_gammaln(a) + matlab_gammaln(b) - matlab_gammaln(a + b)


def betafit(x):
    """The maximum likelihood estimate (a, b) of a beta distribution from data in (0, 1), as
    MATLAB's betafit for data away from 0 and 1: fminsearch on the log parameters from the
    estimates of the geometric means, with TolX and TolFun 1e-6."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    xmin, xmax = x.min(), x.max()
    if xmin < 0 or xmax > 1:
        raise ValueError('betafit needs data in [0, 1]')
    if abs(xmin - xmax) <= 2 * np.spacing(xmax):
        raise ValueError('betafit needs at least two distinct values')
    if not (np.sqrt(np.finfo(float).tiny) <= xmin and xmax <= 1 - np.spacing(1.0) / 2):
        raise ValueError('data at 0 or 1 are not supported')
    n = x.size
    sumlogx = matlab_sum(np.log(x))
    sumlog1mx = matlab_sum(np.log1p(-x))
    tmp1 = np.exp(sumlog1mx / n)
    tmp2 = np.exp(sumlogx / n)
    tmp3 = 1 - tmp1 - tmp2
    pstart = np.log([0.5 * (1 - tmp1) / tmp3, 0.5 * (1 - tmp2) / tmp3])

    def negloglike(p):
        a, b = np.exp(p)
        return n * betaln(a, b) - (a - 1) * sumlogx - (b - 1) * sumlog1mx

    return np.exp(fminsearch(negloglike, pstart, tolx=1e-6, tolf=1e-6)[0])


def betapdf(x, a, b):
    """The beta density at x in (0, 1), as MATLAB's betapdf computes it."""
    x = np.asarray(x, dtype=float)
    logb = np.where(x < 0.1, (b - 1) * np.log1p(-x), (b - 1) * np.log(1 - x))
    return np.exp((a - 1) * np.log(x) + logb - betaln(a, b))


_P1 = [4.945235359296727046734888e0, 2.018112620856775083915565e2, 2.290838373831346393026739e3,
       1.131967205903380828685045e4, 2.855724635671635335736389e4, 3.848496228443793359990269e4,
       2.637748787624195437963534e4, 7.225813979700288197698961e3]
_Q1 = [6.748212550303777196073036e1, 1.113332393857199323513008e3, 7.738757056935398733233834e3,
       2.763987074403340708898585e4, 5.499310206226157329794414e4, 6.161122180066002127833352e4,
       3.635127591501940507276287e4, 8.785536302431013170870835e3]
_P2 = [4.974607845568932035012064e0, 5.424138599891070494101986e2, 1.550693864978364947665077e4,
       1.847932904445632425417223e5, 1.088204769468828767498470e6, 3.338152967987029735917223e6,
       5.106661678927352456275255e6, 3.074109054850539556250927e6]
_Q2 = [1.830328399370592604055942e2, 7.765049321445005871323047e3, 1.331903827966074194402448e5,
       1.136705821321969608938755e6, 5.267964117437946917577538e6, 1.346701454311101692290052e7,
       1.782736530353274213975932e7, 9.533095591844353613395747e6]
_P4 = [1.474502166059939948905062e4, 2.426813369486704502836312e6, 1.214755574045093227939592e8,
       2.663432449630976949898078e9, 2.940378956634553899906876e10, 1.702665737765398868392998e11,
       4.926125793377430887588120e11, 5.606251856223951465078242e11]
_Q4 = [2.690530175870899333379843e3, 6.393885654300092398984238e5, 4.135599930241388052042842e7,
       1.120872109616147941376570e9, 1.488613728678813811542398e10, 1.016803586272438228077304e11,
       3.417476345507377132798597e11, 4.463158187419713286462081e11]
_C = [-1.910444077728e-03, 8.4171387781295e-04, -5.952379913043012e-04, 7.93650793500350248e-04,
      -2.777777777777681622553e-03, 8.333333333333333331554247e-02, 5.7083835261e-03]


def _gammaln1(x):
    """MATLAB's gammaln of one nonnegative double: the algorithm of Cody and Hillstrom (1967)
    as W. J. Cody implemented it, with a table of the logarithms of the factorials."""
    if np.isnan(x):
        return x
    if x < 0:
        return np.nan
    if x > 2.55e305:
        return np.inf
    if x <= np.finfo(float).eps:
        return -np.log(x) if x > 0 else np.inf
    if x <= 0.5:
        xnum, xden = 0.0, 1.0
        for p, q in zip(_P1, _Q1):
            xnum = xnum * x + p
            xden = xden * x + q
        return -np.log(x) + (x * (-5.772156649015328605195174e-1 + x * (xnum / xden)))
    if x < 0.6796875:                       # MATLAB's built-in sends 0.6796875 to the next branch
        xm1 = (x - 0.5) - 0.5
        xnum, xden = 0.0, 1.0
        for p, q in zip(_P2, _Q2):
            xnum = xnum * xm1 + p
            xden = xden * xm1 + q
        return -np.log(x) + xm1 * (4.227843350984671393993777e-1 + xm1 * (xnum / xden))
    if x == np.floor(x) and x <= 100:
        return _LOGFACT[int(x)]
    if x <= 1.5:
        xm1 = (x - 0.5) - 0.5
        xnum, xden = 0.0, 1.0
        for p, q in zip(_P1, _Q1):
            xnum = xnum * xm1 + p
            xden = xden * xm1 + q
        return xm1 * (-5.772156649015328605195174e-1 + xm1 * (xnum / xden))
    if x <= 4:
        xm2 = x - 2
        xnum, xden = 0.0, 1.0
        for p, q in zip(_P2, _Q2):
            xnum = xnum * xm2 + p
            xden = xden * xm2 + q
        return xm2 * (4.227843350984671393993777e-1 + xm2 * (xnum / xden))
    if x <= 12:
        xm4 = x - 4
        xnum, xden = 0.0, -1.0
        for p, q in zip(_P4, _Q4):
            xnum = xnum * xm4 + p
            xden = xden * xm4 + q
        return 1.791759469228055000094023e0 + xm4 * (xnum / xden)
    r = 0.0
    if x <= 2.25e76:
        r = _C[6]
        t = 1 / (x * x)
        for c in _C[:6]:
            r = r * t + c
        r = r / x
    corr = np.log(x)
    return r + 0.9189385332046727417803297 - 0.5 * corr + x * (corr - 1)


def _logfact_table():
    """log((x-1)!) for x = 1, ..., 100, to 18 significant digits as in MATLAB's table."""
    getcontext().prec = 40
    out, lf = [0.0], Decimal(0)
    for x in range(1, 101):
        if x > 1:
            lf += Decimal(x - 1).ln()
        out.append(float(Decimal(f'{lf:.17e}')) if lf else 0.0)
    return out


_LOGFACT = _logfact_table()


def matlab_gammaln(x):
    """MATLAB's gammaln, elementwise."""
    x = np.asarray(x, dtype=float)
    out = np.array([_gammaln1(float(v)) for v in x.ravel()]).reshape(x.shape)
    return out[()] if out.ndim == 0 else out
