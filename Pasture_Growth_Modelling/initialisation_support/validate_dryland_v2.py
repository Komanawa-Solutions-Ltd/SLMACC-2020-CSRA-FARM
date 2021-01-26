"""
 Author: Matt Hanson
 Created: 9/12/2020 11:24 AM
 superceeded by v3
 """
import pandas as pd
import numpy as np
import os
import ksl_env

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from basgra_python import run_basgra_nz
from supporting_functions.plotting import plot_multiple_results
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth
import pandas as pd
import matplotlib.pyplot as plt
from Pasture_Growth_Modelling.initialisation_support.comparison_support import make_mean_comparison, get_horarata_data_old


def calc_past_mean(fun, pg_mode, return_norm=True, freq='month'):
    site = 'oxford'
    mode = 'dryland'
    weather = get_vcsn_record(site=site)
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather, site)
    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

    pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode=pg_mode, resamp_fun=fun, freq=freq))

    out.loc[:, 'pg'] = pg.loc[:, 'pg']
    out.loc[:, 'month'] = pd.Series(out.index).dt.month.values

    out_norm = out.groupby('month').agg({'pg': fun})
    strs = ['2011-{:02d}-15'.format(e) for e in out_norm.index]
    out_norm.loc[:, 'date'] = pd.to_datetime(strs)
    out_norm.set_index('date', inplace=True)

    if return_norm:
        return out_norm
    else:
        return out, out_norm


if __name__ == '__main__':

    data = {
        'hororata': get_horarata_data_old(),
        'basgra_mean': calc_past_mean('mean', 'from_yield')
    }

    plot_multiple_results(data, outdir=None, out_vars=['pg'])

    pass
