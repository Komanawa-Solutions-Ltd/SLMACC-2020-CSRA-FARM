"""
 Author: Matt Hanson
 Created: 27/11/2020 11:11 AM
 """

import pandas as pd
import numpy as np
import os
import ksl_env
from Pasture_Growth_Modelling.storage_parameter_sets import set_store_parameters, get_store_reseed_trig_basal

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from supporting_functions.woodward_2020_params import get_woodward_mean_full_params

default_mode_sites = (
    ('dryland', 'oxford'),
    ('irrigated', 'eyrewell'),
    ('irrigated', 'oxford'),
    ('store400', 'eyrewell'),
    ('store400', 'oxford'),
    ('store600', 'eyrewell'),
    ('store600', 'oxford'),
    ('store800', 'eyrewell'),
    ('store800', 'oxford'),

)  # add storage here!, must have 'store' in mode names

abs_max_irr = 5  # the absolute maximum irrigation values


def get_params_doy_irr(mode, site='eyrewell'):
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

    if mode == 'irrigated' or 'store' in mode:
        # add irrigation parameters
        params['abs_max_irr'] = abs_max_irr
        params['irr_frm_paw'] = 1
        params['IRRIGF'] = 1
        doy_irr = list(range(244, 367)) + list(range(1, 121))  # this will change slightly in leap years

        if 'store' in mode:  # new storage parameterization
            params = set_store_parameters(site, mode, params)
        else:  # historical irrigated parametrisation
            # reseed parameters, set as mean of long term runs in june
            params['reseed_harv_delay'] = 20
            params['reseed_LAI'] = 1.840256e+00
            params['reseed_TILG2'] = 2.194855e+00
            params['reseed_TILG1'] = 4.574009e+00
            params['reseed_TILV'] = 6.949611e+03
            params['reseed_CLV'] = 5.226492e+01
            params['reseed_CRES'] = 9.727732e+00
            params['reseed_CST'] = 1.677470e-01
            params['reseed_CSTUB'] = 0

            # modify inital  # set from start of simulation month (7) mean for the historical period.
            # KEYNOTE worth re-thinking after major change to events
            if site == 'eyrewell':
                params['BASALI'] = 0.747
            elif site == 'oxford':
                params['BASALI'] = 0.723
            else:
                raise ValueError(f'unexpected value for site {site}')

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
        params['BASALI'] = 0.0976  # changed with new calibration from 0.15
        # think about this! average 0.2085 in long term baseline, but I'll keep it at present
        # as it made very little difference in PGR

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


def create_days_harvest(mode, matrix_weather, site, fix_leap=True):
    """
    get the days harvest data
    :param mode: 'dryland' or 'irrigated'
    :return:
    """
    if mode == 'irrigated' or 'store' in mode:
        freq = '15D'  # days
        trig = {m: 1501 for m in range(1, 13)}  # kg harvestable dry matter by month
        targ = {m: 1500 for m in range(1, 13)}  # kg harvestable dry matter by month

        weed_dm_frac = 0
        if site == 'eyrewell':
            # ibasal 0.747
            reseed_trig = 0.696  # 93.172 %
            reseed_basal = 0.727  # 97.32 %

        elif site == 'oxford':
            # ibasal set 0.723
            reseed_trig = 0.678  # 93.77 %
            reseed_basal = 0.704  # 97.37 %

        else:
            raise NotImplementedError()
        if 'store' in mode:
            reseed_trig, reseed_basal = get_store_reseed_trig_basal(site, mode)
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
            3: 0.27,
            4: 0.25,
            5: 0.27,
            6: 0.20,
            7: 0.20,
            8: 0.20,
            9: 0.13,
            10: 0.22,
            11: 0.53,
            12: 0.60,
        }

    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))

    assert (np.in1d(list(range(1, 13)), list(trig.keys())).all() and
            np.in1d(list(range(1, 13)), list(targ.keys())).all()), 'trig and targ must have all months defined'

    if not isinstance(weed_dm_frac, dict):
        weed_dm_frac = {e: weed_dm_frac for e in range(1, 13)}

    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)),  # set filler values
                                 'reseed_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not reseed
                                 'reseed_basal': np.zeros(len(matrix_weather)),  # set filler values
                                 }, index=matrix_weather.index)

    # start harvesting at the same point
    harv_days = pd.Series(pd.date_range(start=matrix_weather.index.min() + pd.DateOffset(days=5),
                                        end=matrix_weather.index.max(), freq=freq))
    if fix_leap:
        # move any harvests that fall on a leap day to the last day in feb.
        idx = (harv_days.dt.month == 2) & (harv_days.dt.day == 29)
        harv_days.loc[idx] = pd.to_datetime([f'{y}-02-28' for y in harv_days[idx].dt.year])

    set_trig = [trig[m] for m in harv_days.dt.month]
    set_targ = [targ[m] for m in harv_days.dt.month]
    days_harvest.loc[harv_days, 'harv_trig'] = set_trig
    days_harvest.loc[harv_days, 'harv_targ'] = set_targ

    # set harvest on last day
    harv_days = days_harvest.index.max()
    days_harvest.loc[harv_days, 'harv_trig'] = trig[harv_days.month]
    days_harvest.loc[harv_days, 'harv_targ'] = targ[harv_days.month]

    # set reseed dates
    days_harvest.loc[days_harvest.doy == 152, 'reseed_trig'] = reseed_trig
    days_harvest.loc[days_harvest.doy == 152, 'reseed_basal'] = reseed_basal

    # set weed fraction
    for m in range(1, 13):
        days_harvest.loc[days_harvest.index.month == m, 'weed_dm_frac'] = weed_dm_frac[m]

    return days_harvest


def create_matrix_weather(mode, weather_data, restriction_data, rest_key='f_rest', fix_leap=True):
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
    if mode == 'irrigated' or 'store' in mode:

        assert (weather_data.index.name ==
                restriction_data.index.name ==
                'date'), 'expected input data to have index of date'
        assert len(weather_data) == len(restriction_data), 'restriction and weather data must be the same length'
        assert (weather_data.index ==
                restriction_data.index).all(), 'restriction data and weather data must have identical dates'
        test_dates = pd.date_range(weather_data.index.min(),
                                   weather_data.index.max())
        if fix_leap:
            test_dates = test_dates[~((test_dates.month == 2) & (test_dates.day == 29))]
            test_dates = pd.to_datetime(
                [f'{y}-{m:02d}-{d:02d}' for y, m, d in zip(test_dates.year, test_dates.month, test_dates.day)])

        assert (weather_data.index == test_dates).all(), 'weather and rest data must not be missing days'

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
        matrix_weather.loc[:, 'irr_trig_store'] = 0
        matrix_weather.loc[:, 'irr_targ_store'] = 0
        matrix_weather.loc[:, 'external_inflow'] = 0
        params, doy_irr = get_params_doy_irr(mode, site='eyrewell')  # just using DOY_irr
        matrix_weather.loc[~np.in1d(matrix_weather.doy, doy_irr), 'max_irr'] = 0


    elif mode == 'dryland':
        assert restriction_data is None, 'restriction data must be None in a dryland scenario'
        assert (weather_data.index.name ==
                'date'), 'expected input data to have index of date'
        test_dates = pd.date_range(weather_data.index.min(),
                                   weather_data.index.max())
        if fix_leap:
            test_dates = test_dates[~((test_dates.month == 2) & (test_dates.day == 29))]
            test_dates = pd.to_datetime(
                [f'{y}-{m:02d}-{d:02d}' for y, m, d in zip(test_dates.year, test_dates.month, test_dates.day)])

        assert (weather_data.index == test_dates).all(), 'weather and rest data must not be missing days'

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
        matrix_weather.loc[:, 'irr_trig_store'] = 0
        matrix_weather.loc[:, 'irr_targ_store'] = 0
        matrix_weather.loc[:, 'external_inflow'] = 0




    else:
        raise ValueError('unexpected mode: {}, values are "irrigated" or "dryland"'.format(mode))
    return matrix_weather
