"""fminbnd, fminsearch and fzero with the algorithms, defaults and stopping rules of
MATLAB's, so that an optimization or a root search here takes the steps of its MATLAB twin:
golden-section search with parabolic interpolation (Forsythe, Malcolm and Moler, 1977), the
Nelder-Mead simplex method as set out by Lagarias, Reeds, Wright and Wright (1998), and the
zero finder of Brent (1973). fminunc stands in for MATLAB's quasi-Newton fminunc on a function
of one variable: the minimizer agrees with MATLAB's within fminunc's tolerance, and the
returned second derivative uses fminunc's finite-difference formula.

See:
Brent, R.P. (1973). Algorithms for Minimization without Derivatives, Prentice-Hall.
Forsythe, G.E., Malcolm, M.A. and Moler, C.B. (1977). Computer Methods for Mathematical
Computations, Prentice-Hall.
Lagarias, J.C., Reeds, J.A., Wright, M.H. and Wright, P.E. (1998). Convergence Properties
of the Nelder-Mead Simplex Method in Low Dimensions, SIAM Journal on Optimization, 9(1):
112-147.
"""

import numpy as np

_SEPS = np.sqrt(np.finfo(float).eps)


def fminbnd(f, ax, bx, tolx=1e-4, maxfun=500, maxiter=500):
    """Minimize the scalar function f on the interval (ax, bx); returns (x, f(x))."""
    if ax > bx:
        raise ValueError('the lower bound exceeds the upper bound')
    c = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = float(ax), float(bx)
    v = a + c * (b - a)
    w = v
    xf = v
    d = 0.0
    e = 0.0
    fx = f(xf)
    funccount = 1
    it = 0
    fv = fx
    fw = fx
    xm = 0.5 * (a + b)
    tol1 = _SEPS * abs(xf) + tolx / 3.0
    tol2 = 2.0 * tol1
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = True
        if abs(e) > tol1:                       # try a parabolic fit
            golden = False
            r = (xf - w) * (fx - fv)
            q = (xf - v) * (fx - fw)
            p = (xf - v) * q - (xf - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = d
            if abs(p) < abs(0.5 * q * r) and p > q * (a - xf) and p < q * (b - xf):
                d = p / q
                x = xf + d
                if (x - a) < tol2 or (b - x) < tol2:
                    d = tol1 * (np.sign(xm - xf) + ((xm - xf) == 0))
            else:
                golden = True
        if golden:
            e = (a - xf) if xf >= xm else (b - xf)
            d = c * e
        x = xf + (np.sign(d) + (d == 0)) * max(abs(d), tol1)
        fu = f(x)
        funccount += 1
        it += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            v, fv = w, fw
            w, fw = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fw or w == xf:
                v, fv = w, fw
                w, fw = x, fu
            elif fu <= fv or v == xf or v == w:
                v, fv = x, fu
        xm = 0.5 * (a + b)
        tol1 = _SEPS * abs(xf) + tolx / 3.0
        tol2 = 2.0 * tol1
        if funccount >= maxfun or it >= maxiter:
            break
    return float(xf), fx


def fminsearch(f, x0, tolx=1e-4, tolf=1e-4, maxfun=None, maxiter=None):
    """Minimize f from the starting point x0 by the Nelder-Mead simplex method; returns
    (x, f(x)), with x shaped like x0. maxfun and maxiter default to 200 times the number
    of unknowns."""
    x0 = np.asarray(x0, dtype=float)
    shape = x0.shape
    xin = x0.reshape(-1, order='F').copy()
    n = xin.size
    maxfun = 200 * n if maxfun is None else maxfun
    maxiter = 200 * n if maxiter is None else maxiter
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5

    def fun(z):
        return f(z.reshape(shape, order='F'))

    v = np.zeros((n, n + 1))
    fv = np.zeros(n + 1)
    v[:, 0] = xin
    fv[0] = fun(xin)
    for j in range(n):                          # the initial simplex
        y = xin.copy()
        y[j] = (1 + 0.05) * y[j] if y[j] != 0 else 0.00025
        v[:, j + 1] = y
        fv[j + 1] = fun(y)
    order = np.argsort(fv, kind='stable')
    fv, v = fv[order], v[:, order]
    itercount = 1
    func_evals = n + 1
    while func_evals < maxfun and itercount < maxiter:
        if (np.max(np.abs(fv[0] - fv[1:])) <= max(tolf, 10 * np.spacing(abs(fv[0])))
                and np.max(np.abs(v[:, 1:] - v[:, [0]]))
                <= max(tolx, 10 * np.spacing(abs(np.max(v[:, 0]))))):
            break
        xbar = np.sum(v[:, :n], axis=1) / n     # centroid of the n best points
        xr = (1 + rho) * xbar - rho * v[:, -1]
        fxr = fun(xr)
        func_evals += 1
        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[:, -1]
            fxe = fun(xe)
            func_evals += 1
            if fxe < fxr:
                v[:, -1], fv[-1] = xe, fxe
            else:
                v[:, -1], fv[-1] = xr, fxr
        elif fxr < fv[n - 1]:
            v[:, -1], fv[-1] = xr, fxr
        else:
            shrink = False
            if fxr < fv[-1]:                    # outside contraction
                xc = (1 + psi * rho) * xbar - psi * rho * v[:, -1]
                fxc = fun(xc)
                func_evals += 1
                if fxc <= fxr:
                    v[:, -1], fv[-1] = xc, fxc
                else:
                    shrink = True
            else:                               # inside contraction
                xcc = (1 - psi) * xbar + psi * v[:, -1]
                fxcc = fun(xcc)
                func_evals += 1
                if fxcc < fv[-1]:
                    v[:, -1], fv[-1] = xcc, fxcc
                else:
                    shrink = True
            if shrink:
                for j in range(1, n + 1):
                    v[:, j] = v[:, 0] + sigma * (v[:, j] - v[:, 0])
                    fv[j] = fun(v[:, j])
                func_evals += n
        order = np.argsort(fv, kind='stable')
        fv, v = fv[order], v[:, order]
        itercount += 1
    return v[:, 0].reshape(shape, order='F'), fv[0]


def fminunc(f, x0):
    """Minimize a function of one variable from x0, in place of MATLAB's fminunc; returns
    (x, f(x), hess), with hess the finite-difference second derivative that fminunc returns,
    from function values only: (f(x + 2h) - 2 f(x + h) + f(x))/h^2, with
    h = eps^(1/4) sign(x) max(|x|, 1). The line search of fminunc is not published, so its
    iterates cannot be reproduced. The minimizer here comes from the Nelder-Mead method at a
    tolerance of 1e-10, so it lies within fminunc's tolerance of 1e-6 of fminunc's."""
    x, fx = fminsearch(lambda z: f(z[0]), np.array([float(x0)]), tolx=1e-10, tolf=1e-10)
    x = float(x[0])
    eps = np.finfo(float).eps
    h = eps**0.25 * np.sign(x) * max(abs(x), 1.0)
    h = np.sign(h + eps) * abs(h)
    f1 = f(x + h)
    f2 = f(x + h + h)
    return x, fx, (f2 - f1 - f1 + fx) / (h * h)


def fzero(f, bracket, tolx=np.finfo(float).eps):
    """A zero of the scalar function f in the interval bracket = (a, b), where f(a) and f(b)
    have opposite signs; returns (x, f(x)). MATLAB's fzero for a given bracket: the method of
    Brent (1973), bisection or interpolation at each step, stopped when the bracket is within
    2*tolx*max(|x|, 1) of the current point."""
    a, b = float(bracket[0]), float(bracket[1])
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError('the function values at the interval end points must be finite')
    if fa == 0.0:
        return a, fa
    if fb == 0.0:
        return b, fb
    if (fa > 0.0) == (fb > 0.0):
        raise ValueError('the function values at the interval end points must differ in sign')
    fc = fb
    c = d = e = 0.0
    while fb != 0 and a != b:
        # b is the best point so far, a the previous one, and c on the other side of the zero
        if (fb > 0) == (fc > 0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        m = 0.5 * (c - b)
        toler = 2.0 * tolx * max(abs(b), 1.0)
        if abs(m) <= toler or fb == 0.0:
            break
        if abs(e) < toler or abs(fa) <= abs(fb):
            d = e = m                               # bisection
        else:
            s = fb / fa
            if a == c:                              # linear interpolation
                p = 2.0 * m * s
                q = 1.0 - s
            else:                                   # inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if 2.0 * p < 3.0 * m * q - abs(toler * q) and p < abs(0.5 * e * q):
                e = d
                d = p / q
            else:
                d = e = m
        a, fa = b, fb
        if abs(d) > toler:
            b = b + d
        elif b > c:
            b = b - toler
        else:
            b = b + toler
        fb = f(b)
    return b, fb
