import numpy as np
from scipy.linalg import solve


def minn(p, c1, c2, c3, Y0, Yt, n0pre):
    """bvt.priors.minn - Minnesota prior constructor for a VAR(p) with intercept.

      beta_Minn, V_Minn, Sig_hat, U_hat = bvt.priors.minn(p, c1, c2, c3, Y0, Yt, n0pre)

      p        : lag length
      c1,c2,c3 : own-lag, cross-lag and intercept shrinkage
      Y0       : presample rows; the last n0pre of them are prepended to Yt for
                 the univariate AR(4) fits, tmpY = vstack([Y0[-n0pre:], Yt])
      Yt       : T x n estimation sample
      n0pre    : number of presample rows to prepend. The AR(4) design matrix is
                 conformable only when the prepended block has exactly 4 rows, so
                 n0pre = 4; n0pre = p runs only for p = 4.
      beta_Minn, V_Minn : prior mean (zeros) and prior variances of alpha, stacked
                 equation by equation, intercept first in each block
      Sig_hat  : the n univariate AR(4) residual variances
      U_hat    : the T x n AR(4) residuals

    Prior: alpha ~ N(beta_Minn, diag(V_Minn)), equation by equation, with
    intercept variance c3, own-lag variance c1/l^2, cross-lag variance
    c2*sig2_i/(l^2*sig2_j); sig2 are univariate AR(4) residual variances.

    See:
    Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds),
    Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham
    Chan, J.C.C. (2023). Comparing Stochastic Volatility Specifications for
    Large Bayesian VARs, Journal of Econometrics, 235(2): 1419-1446.
    """
    Y0 = np.asarray(Y0, dtype=float)
    Yt = np.asarray(Yt, dtype=float)
    p, n0pre = int(p), int(n0pre)
    Tt, n = Yt.shape
    k = 1 + n * p
    beta_Minn = np.zeros(k * n)
    V_Minn = np.zeros(k * n)
    sig2 = np.zeros(n)
    tmpY = np.vstack([Y0[Y0.shape[0] - n0pre:], Yt])
    U_hat = np.zeros((Tt, n))
    for i in range(n):
        Z = np.column_stack([np.ones(Tt), tmpY[3:-1, i], tmpY[2:-2, i], tmpY[1:-3, i],
                             tmpY[:-4, i]])
        tmpb = solve(Z.T @ Z, Z.T @ tmpY[4:, i], assume_a='pos')
        U_hat[:, i] = tmpY[4:, i] - Z @ tmpb
        sig2[i] = np.mean(U_hat[:, i]**2)
    count = 0
    for i in range(n):
        for ii in range(k):
            j = (ii - 1) % n                         # variable index
            lag = -(-ii // n)                        # lag length
            if ii == 0:                              # intercept
                V_Minn[count] = c3
            elif i == j:                             # own lag
                V_Minn[count] = c1 / lag**2
            else:                                    # lag of another variable
                V_Minn[count] = c2 * sig2[i] / (lag**2 * sig2[j])
            count += 1
    return beta_Minn, V_Minn, sig2, U_hat
