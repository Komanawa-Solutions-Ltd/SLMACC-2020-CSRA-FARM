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


default_mode_sites = (
    ('dryland', 'oxford'),
    ('irrigated', 'eyrewell'),
    ('irrigated', 'oxford'),
)

abs_max_irr = 5  # the absolute maximum irrigation values
# todo check initals for SWG data, consider setting to mean of start month of 'average' conditions
def get_params_doy_irr(mode):
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
        doy_irr = list(range(245, 367)) + list(range(1, 122))

        # reseed parameteres, set as mean of long term runs in june
        params['reseed_harv_delay'] = 20
        params['reseed_LAI'] = 1.840256e+00
        params['reseed_TILG2'] = 2.194855e+00
        params['reseed_TILG1'] = 4.574009e+00
        params['reseed_TILV'] = 6.949611e+03
        params['reseed_CLV'] = 5.226492e+01
        params['reseed_CRES'] = 9.727732e+00
        params['reseed_CST'] = 1.677470e-01
        params['reseed_CSTUB'] = 0

        # modify inital  # set from start of simulation month (7) mean
        params['BASALI'] = 0.73843861

        # set from a mid point value not important for percistance, but important to stop inital high yeild!
        # set to start of simulation start month(7) average
        params['LOG10CLVI'] = np.log10(51.998000)
        params['LOG10CRESI'] = np.log10(9.627059)
        params['LOG10CRTI'] = np.log10(125.966156)
    elif mode == 'dryland':
        # add irrigation parameters
        params['irr_frm_paw'] = 1
        params['IRRIGF'] = 0
        doy_irr = [0]

        # modify inital values for dryland
        # set from a mid point value, left as it is part of the dryland 'optimisation' process
        params['BASALI'] = 0.15

        # reseed parameteres, set as mean of long term runs in june
        params['reseed_harv_delay'] = 40
        params['reseed_LAI'] = 1.436825e-01
        params['reseed_TILG2'] = 0
        params['reseed_TILG1'] = 8.329094e-01
        params['reseed_TILV'] = 2.187792e+03
        params['reseed_CLV'] = 3.744545e+00
        params['reseed_CRES'] = 7.042976e-01
        params['reseed_CST'] = 0
        params['reseed_CSTUB'] = 4.305489e-04

        # set from a mid point value not important for persistence, but important to stop initial high yield!,
        # set to start of simulation month (7) average
        params['LOG10CLVI'] = np.log10(3.072320)
        params['LOG10CRESI'] = np.log10(0.586423)
        params['LOG10CRTI'] = np.log10(13.773034)
    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))

    return params, doy_irr


def create_days_harvest(mode, matrix_weather, site):
    """
    get the days harvest data
    :param mode: 'dryland' or 'irrigated'
    :return:
    """
    if mode == 'irrigated':
        freq = '15D'  # days
        trig = {m: 1501 for m in range(1, 13)}  # kg harvestable dry matter by month
        targ = {m: 1500 for m in range(1, 13)}  # kg harvestable dry matter by month

        weed_dm_frac = 0
        if site == 'eyrewell':
            reseed_trig = 0.696
            reseed_basal = 0.727

        elif site == 'oxford':
            reseed_trig = 0.678
            reseed_basal = 0.704

        else:
            raise NotImplementedError()
    elif mode == 'dryland':
        freq = 'M'  # end of each month
        trig = {m: 601 for m in range(4, 12)}  # kg harvestable dry matter
        targ = {m: 600 for m in range(4, 12)}  # kg harvestable dry matter

        # set trig/targ higher for summer months
        trig.update({m: 801 for m in [12, 1, 2, 3]})  # kg harvestable dry matter
        targ.update({m: 800 for m in [12, 1, 2, 3]})  # kg harvestable dry matter

        reseed_trig = 0.059
        reseed_basal = 0.090

        weed_dm_frac = {
            1: 0.42,
            2: 0.30,
            3: 0.33,
            4: 0.32,
            5: 0.46,
            6: 0.59,
            7: 0.70,
            8: 0.57,
            9: 0.33,
            10: 0.37,
            11: 0.62,
            12: 0.62,
        }

    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))

    assert (np.in1d(list(range(1, 13)), list(trig.keys())).all() and
            np.in1d(list(range(1, 13)), list(targ.keys())).all()), 'trig and targ must have all months defined'

    if not isinstance(weed_dm_frac, dict):
        weed_dm_frac = {e: weed_dm_frac for e in range(1, 13)}

    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    dates = pd.to_datetime(strs, format='%Y-%j')
    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)),  # set filler values
                                 'reseed_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not reseed
                                 'reseed_basal': np.zeros(len(matrix_weather)),  # set filler values
                                 })

    # start harvesting at the same point
    harv_days = pd.date_range(start=dates.min() + pd.DateOffset(days=5), end=dates.max(), freq=freq)
    set_trig = [trig[m] for m in harv_days.month]
    set_targ = [targ[m] for m in harv_days.month]
    days_harvest.loc[harv_days, 'harv_trig'] = set_trig
    days_harvest.loc[harv_days, 'harv_targ'] = set_targ

    # set harvest on last day
    harv_days = days_harvest.index.max()
    days_harvest.loc[harv_days, 'harv_trig'] = trig[harv_days.month]
    days_harvest.loc[harv_days, 'harv_targ'] = targ[harv_days.month]

    # set reseed dates
    harv_days = dates.dayofyear == 152  # set to end of june as this is the end of the yield period.
    days_harvest.loc[harv_days, 'reseed_trig'] = reseed_trig
    days_harvest.loc[harv_days, 'reseed_basal'] = reseed_basal

    # set weed fraction
    for m in range(1, 13):
        days_harvest.loc[days_harvest.index.month == m, 'weed_dm_frac'] = weed_dm_frac[m]

    return days_harvest


def create_matrix_weather(mode, weather_data, restriction_data, rest_key='f_rest'):
    """

    :param mode: one of ['irrigated', 'dryland']
    :param weather_data: weather data with datetime index named date, no missing days, and has at least the following
                         keys: ['year', 'doy', 'radn', 'tmin', 'tmax', 'rain', 'pet']
    :param restriction_data: if irrigated then None, otherwise restriction record with datetime index name date with
                             no missing days and at least restriction fraction (rest_key)
    :param rest_key: key to restriction fraction in restriction data
    :return:
    """
    # create from the outputs of greg's work and adds in the irrigation parameters if needed
    if mode == 'irrigated':

        assert (weather_data.index.name ==
                restriction_data.index.name ==
                'date'), 'expected input data to have index of date'
        assert len(weather_data) == len(restriction_data), 'restriction and weather data must be the same length'
        assert (weather_data.index ==
                restriction_data.index).all(), 'restriction data and weather data must have identical dates'
        assert (weather_data.index ==
                pd.date_range(weather_data.index.min(),
                              weather_data.index.max())).all(), 'weather and rest data must not be missing days'

        weather_data = pd.merge(weather_data, restriction_data.loc[:, rest_key], left_index=True, right_index=True)
        matrix_weather = weather_data.loc[:, ['year',
                                              'doy',
                                              'radn',
                                              'tmin',
                                              'tmax',
                                              'rain',
                                              'pet']]

        matrix_weather.loc[:, 'max_irr'] = abs_max_irr * (1 - weather_data.loc[:, rest_key])
        matrix_weather.loc[:, 'irr_trig'] = 0.60
        matrix_weather.loc[:, 'irr_targ'] = 0.75

        # set trig/targ for summer days
        idx = np.in1d(weather_data.loc[:, 'month'], [12, 1, 2])  # set to DOY_IRR
        matrix_weather.loc[idx, 'irr_trig'] = 0.75
        matrix_weather.loc[idx, 'irr_targ'] = 0.90

    elif mode == 'dryland':
        assert restriction_data is None, 'restriction data must be None in a dryland scenario'
        assert (weather_data.index.name ==
                'date'), 'expected input data to have index of date'
        assert (weather_data.index ==
                pd.date_range(weather_data.index.min(),
                              weather_data.index.max())).all(), 'weather and rest data must not be missing days'

        matrix_weather = weather_data.loc[:, ['year',
                                              'doy',
                                              'radn',
                                              'tmin',
                                              'tmax',
                                              'rain',
                                              'pet']]

        matrix_weather.loc[:, 'max_irr'] = 0
        matrix_weather.loc[:, 'irr_trig'] = 0
        matrix_weather.loc[:, 'irr_targ'] = 0
    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))
    return matrix_weather
