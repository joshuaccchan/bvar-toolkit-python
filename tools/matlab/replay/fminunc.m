function [x, fval, exitflag, output, grad, hess] = fminunc(fun, x0, options) %#ok<INUSD>
% fminunc - replay version for the tests: the minimum of a function of one variable as the
% Python port finds it, since the line search of fminunc is not published and its iterates
% cannot be reproduced. fminsearch from x0 at tolerances of 1e-10, then the finite-difference
% second derivative that fminunc returns, (f(x + 2h) - 2 f(x + h) + f(x))/h^2 with
% h = eps^(1/4) sign(x) max(|x|, 1). The Python twin is bvar_toolkit._optim.fminunc.
[x, fval] = fminsearch(fun, x0, optimset('TolX', 1e-10, 'TolFun', 1e-10));
h = eps^(1/4)*sign(x)*max(abs(x), 1);
h = sign(h + eps)*abs(h);
f1 = fun(x + h);
f2 = fun(x + h + h);
hess = (f2 - f1 - f1 + fval)/(h*h);
exitflag = 1;  output = struct();  grad = NaN;
end
