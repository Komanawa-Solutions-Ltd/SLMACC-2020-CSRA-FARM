"""
 Author: Matt Hanson
 Created: 9/12/2020 11:24 AM
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
from Pasture_Growth_Modelling.initialisation_support.inital_long_term_runs import run_past_basgra_irrigated, \
    run_past_basgra_dryland
import pandas as pd
import matplotlib.pyplot as plt


def calc_past_mean(fun, pg_mode, return_norm=True):
    site = 'oxford'
    mode = 'dryland'
    weather = get_vcsn_record(site)
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather)
    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

    pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode=pg_mode, resamp_fun=fun, freq='1d'))

    out.loc[:, 'pg'] = pg.loc[:, 'pg']
    out.loc[:, 'month'] = pd.Series(out.index).dt.month.values

    out_norm = out.groupby(['year', 'month']).agg({'pg': fun,
                                                   'doy': 'mean'}).reset_index().groupby('month').agg({'pg': fun,
                                                                                                       'doy': 'mean'})
    full_out = pd.DataFrame(index=pd.date_range('2011-01-01', '2011-12-31'), columns=['pg'])
    strs = ['2011-{:02d}-15'.format(e) for e in out_norm.index]
    full_out.loc[pd.to_datetime(strs), 'pg'] = out_norm.loc[:, 'pg'].values
    full_out.loc[:, 'pg'] = pd.to_numeric(full_out.loc[:, 'pg']).interpolate(method='linear')
    full_out.loc[:,'doy'] = pd.Series(full_out.index).dt.dayofyear.values
    full_out.set_index('doy', inplace=True)

    if return_norm:
        return full_out
    else:
        return out, full_out


def get_horarata_data():
    out = pd.read_csv(
        ksl_env.shared_drives(r"SLMACC_2020\pasture_growth_modelling\dryland tuning\hororata_dryland.csv"))
    out.loc[:, 'date'] = pd.to_datetime(out.loc[:, 'date'])
    out.loc[:, 'doy'] = out.loc[:, 'date'].dt.dayofyear
    out.set_index('doy', inplace=True)

    full_out = pd.DataFrame(index=range(1, (367) * 3), columns=['pg'])

    idx = out.index.values
    full_out.loc[idx, 'pg'] = out.loc[:, 'pg'].values
    idx += 365
    full_out.loc[idx, 'pg'] = out.loc[:, 'pg'].values
    full_out.loc[idx + 365, 'pg'] = out.loc[:, 'pg'].values
    full_out.loc[:, 'pg'] = pd.to_numeric(full_out.loc[:, 'pg'])
    full_out.loc[:, 'pg'] = full_out.loc[:, 'pg'].interpolate(method='linear')

    idx = np.arange(1, 367)
    full_out = full_out.loc[idx + 365]
    full_out.loc[:, 'doy'] = idx
    full_out.set_index('doy', inplace=True)

    return full_out


if __name__ == '__main__':

    #todo something is weird here... make datetimes, and check against pasture growth deficiit calc...

    data = {
        'hororata': get_horarata_data(),
        'basgra_mean+2 x 10': (calc_past_mean('mean', 'from_dmh')+2) * 10
    }

    plot_multiple_results(data, outdir=None, out_vars=['pg'])

    pass
