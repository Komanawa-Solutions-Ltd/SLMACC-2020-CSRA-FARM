"""
 Author: Matt Hanson
 Created: 25/11/2020 11:42 AM
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


def run_past_basgra_irrigated(return_inputs=False):
    mode = 'irrigated'
    print('running: {}'.format(mode))
    weather = get_vcsn_record()
    rest = get_restriction_record()
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather)

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    if return_inputs:
        return out, (params, doy_irr, matrix_weather, days_harvest)
    return out


def run_past_basgra_dryland(return_inputs=False):
    mode = 'dryland'
    print('running: {}'.format(mode))
    weather = get_vcsn_record()
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather)

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    if return_inputs:
        return out, (params, doy_irr, matrix_weather, days_harvest)
    return out


# todo look at ibasal eyrewell v oxford
# todo look at yeilds
# todo look at percistance.

if __name__ == '__main__':
    data = {'irrigated': run_past_basgra_irrigated(),
            'dryland': run_past_basgra_dryland()}
    plot_multiple_results(data, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'RAIN',
                                          'EVAP',
                                          'TRAN', 'per_PAW'])
