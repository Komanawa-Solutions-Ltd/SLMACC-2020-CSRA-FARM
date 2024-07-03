"""
 Author: Matt Hanson
 Created: 23/11/2020 12:46 PM

 This was an intial approach, moved to looking at the full time period
 """
import pandas as pd
import numpy as np
import os
import project_base
from komanawa.basgra_nz_py.check_basgra_python.support_for_tests import get_lincoln_broadfield
from komanawa.basgra_nz_py.basgra_python import run_basgra_nz
from komanawa.basgra_nz_py.supporting_functions.plotting import plot_multiple_results
from komanawa.basgra_nz_py.supporting_functions.woodward_2020_params import get_woodward_mean_full_params


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

    out = {}
    start_mod1, dates, params, matrix_weather, days_harvest, doy_irr = _base_restriction_data()
    print(start_mod1)

    stop_mod1 = start_mod1 + pd.DateOffset(days=duration)
    start_mod2 = stop_mod1 + pd.DateOffset(days=rest)
    stop_mod2 = start_mod2 + pd.DateOffset(days=duration)
    idx = ((dates < stop_mod1) & (dates >= start_mod1)) | ((dates < stop_mod2) & (dates >= start_mod2))

    out['no_rest'] = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)

    matrix_weather_new = matrix_weather.copy(deep=True)
    matrix_weather_new.loc[idx, 'max_irr'] *= (1 - restrict)

    out['partial_rest'] = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False)

    matrix_weather_new = matrix_weather.copy(deep=True)
    matrix_weather_new.loc[idx, 'max_irr'] = 0
    out['full_rest'] = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False)

    out['mixed_rest'] = out['full_rest'] * restrict + out['no_rest'] * (1 - restrict)

    for k in out:
        out[k].loc[:, 'per_PAW'] = out[k].loc[:, 'PAW'] / out[k].loc[:, 'MXPAW']

    plot_multiple_results(out, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'per_PAW'],
                          outdir=outdir, title_str=ttl_str)


def _base_restriction_data():
    params = get_woodward_mean_full_params('lincoln')
    params['irr_frm_paw'] = 1
    params['IRRIGF'] = 1
    doy_irr = list(range(1,367))

    matrix_weather = get_lincoln_broadfield() # change to section of our data., now is superceeded by long term
    matrix_weather.drop(columns=['rain_def', 'rain_runoff'], inplace=True)
    matrix_weather.loc[:, 'max_irr'] = 5
    matrix_weather.loc[:, 'irr_trig'] = .75
    matrix_weather.loc[:, 'irr_targ'] = .9
    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    org_dates = pd.Series(pd.to_datetime(strs, format='%Y-%j'))
    org_dates.index = matrix_weather.index

    # pull 2 summers worth of data somehow, then set year and DOY to something sensible.
    matrix_weather = matrix_weather.loc[(org_dates >= '2016-12-01') & (org_dates < '2017-03-01')]
    matrix_weather.loc[:, 'org'] = False
    matrix_weather_temp = matrix_weather.copy(deep=True)
    matrix_weather_temp.loc[:, 'org'] = True
    matrix_weather = pd.concat((matrix_weather, matrix_weather_temp)).reset_index()
    matrix_weather.loc[:, 'date'] = pd.Series(pd.date_range(start='2016-07-01', freq='1D', periods=len(matrix_weather)))
    matrix_weather.loc[:, 'year'] = matrix_weather.loc[:, 'date'].dt.year
    matrix_weather.loc[:, 'doy'] = matrix_weather.loc[:, 'date'].dt.dayofyear
    matrix_weather.loc[:, 'rain'] = 0

    start_mod = matrix_weather.loc[matrix_weather.org, 'date'].iloc[0]
    matrix_weather.drop(columns=['org', 'date'], inplace=True)

    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)),  # set filler values
                                 })

    # start harvesting at the same point
    dates = pd.Series(pd.to_datetime(strs, format='%Y-%j'))
    harv_days = pd.Series(pd.date_range(start=dates.iloc[0], end=dates.iloc[-2], freq='{}D'.format(10)))
    idx = np.in1d(dates, harv_days)
    days_harvest.loc[idx, 'harv_trig'] = 1501
    days_harvest.loc[idx, 'harv_targ'] = 1500

    return start_mod, dates, params, matrix_weather, days_harvest, doy_irr


# I get a chunk of data (1 year) # append a non-restriction chuck infront of it, run a


def _temp_basgra_run():
    start_mod, dates, params, matrix_weather, days_harvest, doy_irr = _base_restriction_data()
    print(start_mod)
    temp = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out = {'temp': temp}
    temp.to_csv(r"C:\Users\Matt Hanson\Downloads\test_get_time.csv")
    plot_multiple_results(out, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'PAW'])


if __name__ == '__main__':
    irrigation_restrictions(duration=10,
                            restrict=1,
                            rest=10,
                            outdir=None,
                            ttl_str='')