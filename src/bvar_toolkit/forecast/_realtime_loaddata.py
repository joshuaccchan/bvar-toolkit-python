import numpy as np


def _quarterly(x):
    """Monthly observations averaged to quarters, the incomplete last quarter dropped."""
    n_q = x.size // 3
    return x[:3 * n_q].reshape(n_q, 3).mean(axis=1)


def realtime_loaddata(rt_data, nonrev_data, t, T0, tcode, var_type):
    """bvt.forecast.realtime_loaddata - assembles the vintage-t real-time estimation panel.

      data_t, data_tpk = bvt.forecast.realtime_loaddata(rt_data, nonrev_data, t, T0,
                                                        tcode, var_type)

    rt_data     - dict of real-time vintage matrices, keys 'var1' ... 'varn', one
                  column per vintage
    nonrev_data - dict of non-revised series, same keys (read only for
                  var_type 4)
    t           - vintage index; the estimation panel ends at t-1
    T0          - period of the first vintage (the vintage column is t-T0+1 for
                  quarterly vintages, (t-T0)*3+3 for monthly ones, counted from 1)
    tcode       - n transformation codes: 5 = 400*log-difference, 1 = level
    var_type    - n vintage/observation frequency codes: 1 = quarterly
                  vintage, quarterly observations; 2 = quarterly vintage,
                  monthly observations averaged to quarters; 3 = monthly
                  vintage, monthly observations averaged to quarters;
                  4 = non-revised, monthly observations averaged to quarters
    data_t      - (t-2) x n estimation panel as seen in vintage t
    data_tpk    - 5 x n realized values from the LAST vintage, periods t-1 to t+3

    The periods t, T0 and the vintage columns count from 1, as the periods of the data
    do in the real-time files.

    See:
    Chan, J.C.C. (2020). Large Bayesian Vector Autoregressions. In: P. Fuleky (Eds),
    Macroeconomic Forecasting in the Era of Big Data, 95-125, Springer, Cham.
    """
    n = len(var_type)
    data_t = np.zeros((t - 2, n))
    data_tpk = np.zeros((5, n))                  # actual observations from the last vintage
    for i_var in range(n):
        var_name = f'var{i_var + 1}'
        vt = var_type[i_var]
        if vt == 1:                              # Q vin, Q obs
            allvin = np.asarray(rt_data[var_name], dtype=float)
            tmp_var = allvin[:, t - T0]
            last_vin = allvin[:, -1]
        elif vt == 2:                            # Q vin, M obs
            allvin = np.asarray(rt_data[var_name], dtype=float)
            tmp_var = _quarterly(allvin[:, t - T0])
            last_vin = _quarterly(allvin[:, -1])
        elif vt == 3:                            # M vin, M obs
            allvin = np.asarray(rt_data[var_name], dtype=float)
            tmp_var = _quarterly(allvin[:, (t - T0) * 3 + 2])
            last_vin = _quarterly(allvin[:, -1])
        elif vt == 4:                            # non-revised, M obs
            tmp_var = _quarterly(np.asarray(nonrev_data[var_name], dtype=float).ravel())
            last_vin = tmp_var
        if tcode[i_var] == 5:
            data_t[:, i_var] = 400 * np.log(tmp_var[1:t - 1] / tmp_var[:t - 2])
            y_last_vin = 400 * np.log(last_vin[1:] / last_vin[:-1])
            data_tpk[:, i_var] = y_last_vin[t - 2:t + 3]
        elif tcode[i_var] == 1:
            data_t[:, i_var] = tmp_var[:t - 2]
            data_tpk[:, i_var] = last_vin[t - 2:t + 3]
    return data_t, data_tpk
