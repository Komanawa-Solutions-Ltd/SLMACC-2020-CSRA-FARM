"""
 Author: Matt Hanson
 Created: 27/11/2020 11:11 AM
 """

import pandas as pd
import numpy as np
import os
import ksl_env

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from supporting_functions.woodward_2020_params import get_woodward_mean_full_params


def get_params(mode):
    """
    get the parameter sets for all of the basgra modelling
    :param mode: 'dryland','irrigated'
    :return:
    """

    params = get_woodward_mean_full_params('lincoln')

    # add soil parameters
    params['FWCWP'] = 0.325  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW
    params['FWCFC'] = 0.735  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW
    params['WCST'] = 0.40  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW
    params['BD'] = 1.09  # from smap Wakanui_6a.1 top soil density

    # add harvest parameters
    params['fixed_removal'] = 0
    params['opt_harvfrin'] = 1

    if mode == 'irrigated':
        # add irrigation parameters
        params['irr_frm_paw'] = 1
        params['IRRIGF'] = 1
        params['doy_irr_start'] = 0  # todo update
        params['doy_irr_end'] = 366  # todo update
    elif mode == 'dryland':
        # add irrigation parameters
        params['irr_frm_paw'] = 1
        params['IRRIGF'] = 0
        params['doy_irr_start'] = 0
        params['doy_irr_end'] = 0

        # modify inital values for dryland
        # set from a mid point value
        params['BASALI'] = 0.1  # todo

        # set from a mid point value not important for percistance, but important to stop inital high yeild!
        params['LOG10CLVI'] = np.log10(4.2)  # todo
        params['LOG10CRESI'] = np.log10(0.8)  # todo
        params['LOG10CRTI'] = np.log10(36)  # todo
    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))

    return params


def create_days_harvest(mode, matrix_weather):  # todo
    """
    get the days harvest data
    :param mode: 'dryland' or 'irrigated'
    :return:
    """
    if mode == 'irrigated':
        trig = 1501  # kg harvestable dry matter
        targ = 1500  # kg harvestable dry matter
        freq = 10  # days
    elif mode == 'dryland':  # todo finalize
        trig = 601  # kg harvestable dry matter
        targ = 600  # kg harvestable dry matter
        freq = 25  # days
    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))

    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    dates = pd.to_datetime(strs, format='%Y-%j')
    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)),  # set filler values
                                 })

    # start harvesting at the same point
    harv_days = pd.date_range(start=dates.min() + pd.DateOffset(days=5), end=dates.max(), freq='{}D'.format(freq))
    idx = np.in1d(dates, harv_days)
    days_harvest.loc[idx, 'harv_trig'] = trig
    days_harvest.loc[idx, 'harv_targ'] = targ


# todo check everything
def create_matrix_weather():
    # create from the outputs of greg's work and adds in the irrigation parameters if needed
    raise NotImplementedError
