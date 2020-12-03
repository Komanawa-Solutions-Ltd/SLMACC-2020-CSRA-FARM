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


def run_past_basgra_irrigated(return_inputs=False, site='eyrewell', reseed=True):
    mode = 'irrigated'
    print('running: {}, {}, {}'.format(mode, site, reseed))
    weather = get_vcsn_record(site)
    rest = get_restriction_record()
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather)

    if not reseed:
        days_harvest.loc[:, 'reseed_trig'] = -1

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    if return_inputs:
        return out, (params, doy_irr, matrix_weather, days_harvest)
    return out


def run_past_basgra_dryland(return_inputs=False, site='eyrewell', reseed=True):
    mode = 'dryland'
    print('running: {}, {}, reseed: {}'.format(mode, site, reseed))
    weather = get_vcsn_record(site)
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather)
    if not reseed:
        days_harvest.loc[:, 'reseed_trig'] = -1

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    if return_inputs:
        return out, (params, doy_irr, matrix_weather, days_harvest)
    return out


# todo look at ibasal eyrewell v oxford
# todo look at yeilds
# todo look at percistance.

if __name__ == '__main__':
    data = {
        # 'irrigated_eyrewell': run_past_basgra_irrigated(),
        'dryland_eyrewell_reseed': run_past_basgra_dryland(),
        'dryland_eyrewell': run_past_basgra_dryland(reseed=False),
        'dryland_oxford_reseed': run_past_basgra_dryland(site='oxford'),
        'dryland_oxford': run_past_basgra_dryland(site='oxford', reseed=False),
    }
    out_vars = ['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'RAIN', 'EVAP', 'TRAN', 'per_PAW']
    out_vars = ['per_PAW', 'DMH_RYE', 'DM_RYE_RM', 'YIELD',
                'CLV', 'CRES', 'CST', 'CSTUB', 'CLVD', 'LAI',
                'PHEN', 'TILG2', 'TILG1', 'TILV', 'BASAL', 'RESEEDED']  # for working through reseed
    plot_multiple_results(data, out_vars=out_vars,marker='o')
