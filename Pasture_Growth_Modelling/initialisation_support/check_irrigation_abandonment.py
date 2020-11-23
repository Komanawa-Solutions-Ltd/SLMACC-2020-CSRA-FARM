"""
 Author: Matt Hanson
 Created: 23/11/2020 12:46 PM
 """
import sys

sys.path.append('C:/Users/Matt Hanson/python_projects/BASGRA_NZ_PY')
import pandas as pd
import numpy as np
import os
import ksl_env
from check_basgra_python.support_for_tests import get_lincoln_broadfield, get_woodward_weather, _clean_harvest
from basgra_python import run_basgra_nz
from check_basgra_python.support_for_tests import establish_org_input
from supporting_functions.plotting import plot_multiple_results


def irrigation_restrictions(duration, restrict, rest, outdir, ttl_str):
    """
    plot up irrigatino restriction handling
    run two irritation restrictions with a rest period
    :param duration: length of restriction
    :param restrict: restiction 0-1
    :param rest: non-restriction time in between
    :param outdir: passed to plotting function
    :param ttl_str: title string passed to plotting function
    :return:
    """
    # todo check

    out = {}
    start_mod1, dates, params, matrix_weather, days_harvest = _base_restriction_data()

    stop_mod1 = start_mod1 + pd.DateOffset(days=duration)
    start_mod2 = stop_mod1 + pd.DateOffset(days=rest)
    stop_mod2 = start_mod2 + pd.DateOffset(days=duration)
    idx = (dates < stop_mod1 & dates >= start_mod1) | (dates < stop_mod2 & dates >= start_mod2)

    out['no_rest'] = run_basgra_nz(params, matrix_weather, days_harvest, verbose=False)

    matrix_weather_new = matrix_weather.copy(deep=True)
    matrix_weather.loc[idx, 'max_irr'] *= (1 - restrict)

    out['partial_rest'] = run_basgra_nz(params, matrix_weather_new, days_harvest, verbose=False)

    matrix_weather_new = matrix_weather.copy(deep=True)
    matrix_weather.loc[idx, 'max_irr'] = 0
    out['full_rest'] = run_basgra_nz(params, matrix_weather_new, days_harvest, verbose=False)

    out['mixed_rest'] = out['full_rest'] * restrict + out['no_rest'] * (1 - restrict)

    plot_multiple_results(out, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM'],
                          outdir=outdir, title_str=ttl_str)


def _base_restriction_data():
    params, matrix_weather, days_harvest = establish_org_input('lincoln')
    params['irr_frm_paw'] = 1

    matrix_weather = get_lincoln_broadfield()
    matrix_weather.loc[:, 'max_irr'] = 5
    matrix_weather.loc[:, 'irr_trig'] = .5
    matrix_weather.loc[:, 'irr_targ'] = .9

    # todo pull 2 summers worth of data somehow, then set year and DOY to something sensible.
    # todo return the start mod,


    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)),  # set filler values
                                 })

    # start harvesting at the same point
    dates = pd.to_datetime(strs, format='%Y-%j')
    harv_days = pd.date_range(start=dates.iloc[0], end=dates.iloc[-2], freq='{}D'.format(10))
    idx = np.in1d(dates, harv_days)
    days_harvest.loc[idx, 'harv_trig'] = 1501
    days_harvest.loc[idx, 'harv_targ'] = 1500


    raise NotImplementedError
    return start_mod, dates, params, matrix_weather, days_harvest

# I get a chunk of data (1 year) # append a non-restriction chuck infront of it, run a
