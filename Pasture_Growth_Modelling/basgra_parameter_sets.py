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

        # reseed parameteres # todo set?
        params['reseed_harv_delay'] = 10
        params['reseed_LAI'] = -1
        params['reseed_TILG2'] = -1
        params['reseed_TILG1'] = -1
        params['reseed_TILV'] = -1

        # modify inital
        # params['BASALI'] = 0.1  # todo

        # set from a mid point value not important for percistance, but important to stop inital high yeild!
        # params['LOG10CLVI'] = np.log10(4.2)  # todo
        # params['LOG10CRESI'] = np.log10(0.8)  # todo
        # params['LOG10CRTI'] = np.log10(36)  # todo
    elif mode == 'dryland':
        # add irrigation parameters
        params['irr_frm_paw'] = 1
        params['IRRIGF'] = 0
        doy_irr = [0]

        # modify inital values for dryland
        # set from a mid point value
        params['BASALI'] = 0.1  # todo

        # reseed parameteres
        params['reseed_harv_delay'] = 10
        params['reseed_LAI'] = 0.2 #todo just playing, but seems important, consider setting to average at this time of year
        params['reseed_TILG2'] = -1 #todo just playing, but always 0 at this time of year
        params['reseed_TILG1'] = 0.01 #todo just playing, but might be important
        params['reseed_TILV'] = 150 #todo just playing, but seems important

        # set from a mid point value not important for percistance, but important to stop inital high yeild!
        params['LOG10CLVI'] = np.log10(4.2)  # todo
        params['LOG10CRESI'] = np.log10(0.8)  # todo
        params['LOG10CRTI'] = np.log10(36)  # todo
    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))

    return params, doy_irr


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
        weed_frac = 0
        reseed_trig = -1 #todo?
        reseed_basal = 1 #todo?
    elif mode == 'dryland':  # todo finalize
        trig = 601  # kg harvestable dry matter
        targ = 600  # kg harvestable dry matter
        freq = 25  # days
        weed_frac = 0.0  # todo when dryland work finishes, is this better or just take the fraction anaumoly to eliminate bias
        reseed_trig = 0.06 #todo
        reseed_basal = 0.1 #todo
    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))

    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    dates = pd.to_datetime(strs, format='%Y-%j')
    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)) + weed_frac,  # set filler values
                                 'reseed_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not reseed
                                 'reseed_basal': np.zeros(len(matrix_weather)),  # set filler values
                                 })

    # start harvesting at the same point
    harv_days = pd.date_range(start=dates.min() + pd.DateOffset(days=5), end=dates.max(), freq='{}D'.format(freq))
    idx = np.in1d(dates, harv_days)
    days_harvest.loc[idx, 'harv_trig'] = trig
    days_harvest.loc[idx, 'harv_targ'] = targ

    # set harvest on last day
    idx = days_harvest.index.max()
    days_harvest.loc[idx, 'harv_trig'] = trig
    days_harvest.loc[idx, 'harv_targ'] = targ

    # set reseed dates
    idx = dates.dayofyear == 152  #todo not confirmed, this is when harvest swiches over
    days_harvest.loc[idx, 'reseed_trig'] = reseed_trig
    days_harvest.loc[idx, 'reseed_basal'] = reseed_basal

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
    if mode == 'irrigated':  # todo

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

        matrix_weather.loc[:, 'max_irr'] = 5 * (1 - weather_data.loc[:, rest_key])
        matrix_weather.loc[:, 'irr_trig'] = 0.60
        matrix_weather.loc[:, 'irr_targ'] = 0.75

        # set trig/targ for summer days
        idx = np.in1d(weather_data.loc[:, 'month'], [12, 1, 2])
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
