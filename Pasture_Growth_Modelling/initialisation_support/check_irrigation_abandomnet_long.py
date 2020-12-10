"""
 Author: Matt Hanson
 Created: 4/12/2020 10:20 AM
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

mode = 'irrigated'


def create_irrigation_abandomnet_data(base_name, params, reseed_trig=-1, reseed_basal=1, site='eyrewell'):
    out = {}
    weather = get_vcsn_record(site)
    rest = get_restriction_record()
    p, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    restrict = 1 - matrix_weather.loc[:, 'max_irr'] / 5

    matrix_weather.loc[:, 'max_irr'] = 5
    days_harvest = create_days_harvest(mode, matrix_weather, site)

    # set reseed days harvest
    idx = days_harvest.doy == 152
    days_harvest.loc[idx, 'reseed_trig'] = reseed_trig
    days_harvest.loc[idx, 'reseed_basal'] = reseed_basal

    # run models

    matrix_weather_new = matrix_weather.copy(deep=True)
    matrix_weather_new.loc[restrict >= 0.9999, 'max_irr'] = 0
    temp = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False)
    temp.loc[:, 'f_rest'] = 1 - matrix_weather_new.loc[:, 'max_irr'] / 5
    temp.loc[:, 'pgr'] = calc_pasture_growth(temp, days_harvest, 'from_yeild_regular', '1D', resamp_fun='mean')

    out['{}_no_rest'.format(base_name)] = temp

    matrix_weather_new = matrix_weather.copy(deep=True)
    matrix_weather_new.loc[:, 'max_irr'] *= (1 - restrict)

    temp = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False)
    temp.loc[:, 'f_rest'] = 2 - matrix_weather_new.loc[:, 'max_irr'] / 5
    temp.loc[:, 'pgr'] = calc_pasture_growth(temp, days_harvest, 'from_yeild_regular', '1D', resamp_fun='mean')

    out['{}_partial_rest'.format(base_name)] = temp

    matrix_weather_new = matrix_weather.copy(deep=True)
    matrix_weather_new.loc[:, 'max_irr'] *= (restrict <= 0).astype(int)

    temp = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False)
    temp.loc[:, 'f_rest'] = 3 - matrix_weather_new.loc[:, 'max_irr'] / 5
    temp.loc[:, 'pgr'] = calc_pasture_growth(temp, days_harvest, 'from_yeild_regular', '1D', resamp_fun='mean')
    out['{}_full_rest'.format(base_name)] = temp

    out['{}_mixed_rest'.format(base_name)] = (out['{}_full_rest'.format(base_name)].transpose() *
                                              restrict.values +
                                              out['{}_no_rest'.format(base_name)].transpose() *
                                              (1 - restrict.values)).transpose()

    # paddock level restrictions
    levels = np.arange(5, 110, 5) / 100
    temp_out = []
    for ll, lu in zip(levels[0:-1], levels[1:]):
        matrix_weather_new = matrix_weather.copy(deep=True)

        matrix_weather_new.loc[restrict <= ll, 'max_irr'] = 5
        matrix_weather_new.loc[restrict >= lu, 'max_irr'] = 0
        idx = (restrict > ll) & (restrict < lu)
        matrix_weather_new.loc[idx, 'max_irr'] = 5 * ((restrict.loc[idx] - ll) / 0.05)

        temp = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False)
        temp_out.append(temp.values)
    temp = np.mean(temp_out, axis=0)

    temp2 = out['{}_mixed_rest'.format(base_name)].copy(deep=True).drop(columns=['f_rest', 'pgr'])
    temp2.loc[:, :] = temp
    temp2.loc[:, 'f_rest'] = restrict + 3
    temp2.loc[:, 'pgr'] = calc_pasture_growth(temp2, days_harvest, 'from_yeild_regular', '1D', resamp_fun='mean')

    out['{}_paddock_rest'.format(base_name)] = temp2

    for k in out:
        out[k].loc[:, 'per_PAW'] = out[k].loc[:, 'PAW'] / out[k].loc[:, 'MXPAW']

    return out


if __name__ == '__main__':
    #todo run some stats on paddock vs poor irrigation
    params, doy = get_params_doy_irr(mode)
    out = create_irrigation_abandomnet_data('baseline', params, reseed_trig=0.65, reseed_basal=0.70, site='eyrewell')
    out_vars = ['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'per_PAW', 'pgr', 'f_rest']
    plot_multiple_results(out, out_vars=out_vars, rolling=30, main_kwargs={'alpha': 0.2}, label_main=False,
                          label_rolling=True)
