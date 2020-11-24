"""
 Author: Matt Hanson
 Created: 25/11/2020 8:52 AM
 work through a validation of basgra for dryland crops at lincoln
 data from mills et all 2015
 https://drive.google.com/open?id=1H1Bo-CFrmZcNbLt9kzyVTOzoNB7DE-Zp&authuser=matt%40komanawa.com&usp=drive_fs
 """

import ksl_env
import pandas as pd
import numpy as np
import os
from Climate_Shocks.note_worthy_events.vcsn_pull import vcsn_pull_single_site

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from check_basgra_python.support_for_tests import get_lincoln_broadfield, establish_org_input, _clean_harvest
from basgra_python import run_basgra_nz
from supporting_functions.plotting import plot_multiple_results
from supporting_functions.woodward_2020_params import get_woodward_mean_full_params
from input_output_keys import matrix_weather_keys_pet

backed_dir = ksl_env.shared_drives("SLMACC_2020\pasture_growth_modelling\dryland tuning")
unbacked_dir = ksl_env.mh_unbacked("SLMACC_2020\pasture_growth_modelling\dryland tuning")

if not os.path.exists(backed_dir):
    os.makedirs(backed_dir)

if not os.path.exists(unbacked_dir):
    os.makedirs(unbacked_dir)


def get_org_data():
    raise NotImplementedError


def get_param_set(BASALI=0.25):
    params = get_woodward_mean_full_params('lincoln')
    params['FWCWP'] = 0.40  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW
    params['FWCFC'] = 0.80  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW
    params['WCST'] = 0.46  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW
    params['BD'] = 1.22  # from smap Wakanui_6a.1
    params['fixed_removal'] = 0
    params['opt_harvfrin'] = 1
    params['IRRIGF'] = 0
    params['BASALI'] = BASALI  # todo id correct start and replace
    params['LOG10CLVI'] = np.log10(4.2)  # set from a mid point value  # todo not important for percistance, but important to stop inital high yeild!
    params['LOG10CRESI'] = np.log10(0.8)  # set from a mid point value  # todo not important for percistance, but important to stop inital high yeild!
    params['LOG10CRTI'] = np.log10(36)  # set from a mid point value  # todo not important for percistance, but important to stop inital high yeild!
    # params['HARVFRD'] = 0 # todo not importnat for percistance

    return params


def get_weather_data():

    # below is the lincoln farm, ryegrass percistance is not likely.
    #vcsn_data, use_coords = vcsn_pull_single_site(lat=-43.6333, lon=172.4666, year_min=2002, year_max=2011, use_vars='all')

    # below is the oxford area, to test percistance #todo oxford are percistance, percistance looks possible in oxford
    vcsn_data, use_coords = vcsn_pull_single_site(lat=-43.298068, lon=172.197276, year_min=2002, year_max=2011, use_vars='all')

    # below is the percistance at eyrewell # todo is the presitance possible at eyrewell, seems so, check for full period!
    vcsn_data, use_coords = vcsn_pull_single_site(lat=-43.355787, lon=172.324873, year_min=2002, year_max=2011, use_vars='all')


    vcsn_data.to_csv(os.path.join(unbacked_dir,'vcsn.csv'))

    vcsn_data.rename(columns={'evspsblpot': 'pet', 'pr': 'rain',
                              'rsds': 'radn', 'tasmax': 'tmax', 'tasmin': 'tmin'}, inplace=True)
    vcsn_data.loc[:, 'max_irr'] = 10
    vcsn_data.loc[:, 'irr_trig'] = 0
    vcsn_data.loc[:, 'irr_targ'] = 1

    vcsn_data = vcsn_data.loc[(vcsn_data.date >= '2002-05-03') & (vcsn_data.date <= '2011-06-27')]

    matrix_weather = vcsn_data.loc[:, list(matrix_weather_keys_pet)]


    return matrix_weather


def get_harvest_data(weed_dm_frac, matrix_weather, harv_trig, harv_targ, freq):
    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)) + weed_dm_frac,
                                 })

    # start harvesting at the same point
    dates = pd.Series(pd.to_datetime(strs, format='%Y-%j'))
    harv_days = pd.Series(pd.date_range(start=dates.iloc[0], end=dates.iloc[-2], freq='{}D'.format(freq)))
    idx = np.in1d(dates, harv_days)
    days_harvest.loc[idx, 'harv_trig'] = harv_trig
    days_harvest.loc[idx, 'harv_targ'] = harv_targ
    return days_harvest


def get_input_data(basali, weed_dm_frac, harv_trig, harv_targ, freq):
    params = get_param_set(basali)
    matrix_weather = get_weather_data()
    days_harvest = get_harvest_data(weed_dm_frac=weed_dm_frac, matrix_weather=matrix_weather, harv_trig=harv_trig, harv_targ=harv_targ,freq=freq)
    return params, matrix_weather, days_harvest


def run_inital_basgra(basali, weed_dm_frac, harv_targ, harv_trig, freq):
    """
    run an intial test
    :param basali:
    :param weed_dm_frac:
    :return:
    """
    params, matrix_weather, days_harvest = get_input_data(basali, weed_dm_frac, harv_targ=harv_targ,
                                                          harv_trig=harv_trig, freq=freq)

    temp = run_basgra_nz(params, matrix_weather, days_harvest, verbose=False)
    out = {'temp': temp}
    temp.to_csv(r"C:\Users\Matt Hanson\Downloads\test_get_time.csv")
    plot_multiple_results(out, out_vars=['DM', 'YIELD', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'PAW', 'DMH','BASAL'])

def run_nonirr_lincoln_low_basil(IBASAL, weed, harv_trig, harv_targ, freq):
    params, matrix_weather, days_harvest = establish_org_input('lincoln')

    params['FWCWP'] = 0.40  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW #todo dadb validate dryland
    params['FWCFC'] = 0.80  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW #todo dadb validate dryland
    params['WCST'] = 0.46  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW #todo dadb validate dryland
    params['BD'] = 1.22  # from smap Wakanui_6a.1 #todo dadb validate dryland
    params['fixed_removal'] = 0 #todo dadb validate dryland
    params['opt_harvfrin'] = 1 #todo dadb validate dryland
    params['IRRIGF'] = 0 #todo dadb validate dryland
    params['LOG10CLVI'] = np.log10(4.2)
    params['LOG10CRESI'] = np.log10(0.8)
    params['LOG10CRTI'] = np.log10(36)
    params['CSTI']
    params['LOG10LAII']
    params['PHENI']
    params['TILTOTI']
    params['FRTILGI']
    params['LT50I']

    matrix_weather = get_lincoln_broadfield()
    matrix_weather.loc[:, 'max_irr'] = 10
    matrix_weather.loc[:, 'irr_trig'] = 0
    matrix_weather.loc[:, 'irr_targ'] = 1

    matrix_weather = matrix_weather.loc[:, matrix_weather_keys_pet]

    params['IRRIGF'] = 0  # no irrigation
    params['doy_irr_start'] = 305  # start irrigating in Nov
    params['doy_irr_end'] = 90  # finish at end of march
    params['BASALI'] = IBASAL  # start at 20% basal

    days_harvest = _clean_harvest(days_harvest, matrix_weather)
    days_harvest = get_harvest_data(weed, matrix_weather, harv_trig, harv_targ, freq)

    out = run_basgra_nz(params, matrix_weather, days_harvest, verbose=False)
    out.loc[:,'per_fc'] = out.loc[:,'WAL']/out.loc[:,'WAFC']
    out.loc[:,'per_paw'] = out.loc[:,'PAW']/out.loc[:,'MXPAW']
    temp = run_basgra_nz(params, matrix_weather, days_harvest, verbose=False)
    out = {'temp': temp}
    temp.to_csv(os.path.join(unbacked_dir, 'test_ibasal_out.csv'))
    pd.Series(params).to_csv(os.path.join(unbacked_dir, 'test_ibasal_params.csv'))
    plot_multiple_results(out, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'PAW', 'DMH'])



if __name__ == '__main__':
    run_inital_basgra(basali=0.10, #todo dryland species do not seem stable under this optimisation... are they re-seeded?
                      weed_dm_frac=.10,
                      harv_targ=600,
                      harv_trig=601,
                      freq=10)
    #run_nonirr_lincoln_low_basil(IBASAL=0.3, weed=0.15, harv_trig=1001, harv_targ=1000, freq=25)
