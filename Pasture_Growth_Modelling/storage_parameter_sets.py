"""
 Author: Matt Hanson
 Created: 10/09/2021 10:01 AM
 """
import copy

import numpy as np

#  data at: https://docs.google.com/spreadsheets/d/1-_ZU9wCvGYjn1j2Z_m9ybtziW_fLWHnfFqLd7CkPN5s/edit?usp=sharing

site_mode_dep_params = {
    # ('ex_site', 'ex_mode'): {  # just an example of what is needed, should never be callable
    #    # site, mode specific parameters,
    #    'h2o_store_max_vol': 10000,
    #    # reseed parameteres, set as mean of long term runs in june,
    #    'reseed_harv_delay': 20,

    #    'reseed_LAI': 1.840256e+00,
    #    'reseed_TILG2': 2.194855e+00,
    #    'reseed_TILG1': 4.574009e+00,
    #    'reseed_TILV': 6.949611e+03,
    #    'reseed_CLV': 5.226492e+01,
    #    'reseed_CRES': 9.727732e+00,
    #    'reseed_CST': 1.677470e-01,
    #    'reseed_CSTUB': 0,
    #
    #    # set from a mid point value not important for percistance, but important to stop inital high yeild!,
    #    # set to start of simulation start month(7) average,
    #    'LOG10CLVI': np.log10(51.998000),
    #    'LOG10CRESI': np.log10(9.627059),
    #    'LOG10CRTI': np.log10(125.966156),
    #
    # },

    ('eyrewell', 'store400'): {  # just an example of what is needed, should never be callable
        # site, mode specific parameters,
        'h2o_store_max_vol': 400,
        # reseed parameteres, set as mean of long term runs in june,
        'reseed_harv_delay': 20,

        'reseed_LAI': 1.8883760908499,
        'reseed_TILG2': 0.0823070625,
        'reseed_TILG1': 4.95081235310532,
        'reseed_TILV': 7471.05152692203,
        'reseed_CLV': 50.6370444790093,
        'reseed_CRES': 9.43917193131147,
        'reseed_CST': 0.006359656443429,
        'reseed_CSTUB': 0.000222303625703,

        # set from a mid point value not important for percistance, but important to stop inital high yeild!,
        # set to start of simulation start month(7) average,
        'LOG10CLVI': np.log10(50.6370444790093),
        'LOG10CRESI': np.log10(9.43917193131147),
        'LOG10CRTI': np.log10(126.743297976588),

    },

    ('oxford', 'store400'): {  # just an example of what is needed, should never be callable
        # site, mode specific parameters,
        'h2o_store_max_vol': 400,

        # reseed parameteres, set as mean of long term runs in june,
        'reseed_harv_delay': 20,

        'reseed_LAI': 1.80394541555861,
        'reseed_TILG2': 0,
        'reseed_TILG1': 2.90767378665051,
        'reseed_TILV': 7477.73023434671,
        'reseed_CLV': 45.4330221297619,
        'reseed_CRES': 8.76388967968129,
        'reseed_CST': 0,
        'reseed_CSTUB': 1.90270560680405E-11,

        # set from a mid point value not important for percistance, but important to stop inital high yeild!,
        # set to start of simulation start month(7) average,
        'LOG10CLVI': np.log10(45.4330221297619),
        'LOG10CRESI': np.log10(8.76388967968129),
        'LOG10CRTI': np.log10(159.193170542361),

    },

    ('eyrewell', 'store600'): {  # just an example of what is needed, should never be callable
        # site, mode specific parameters,
        'h2o_store_max_vol': 600,
        # reseed parameteres, set as mean of long term runs in june,
        'reseed_harv_delay': 20,

        'reseed_LAI': 1.89829178077739,
        'reseed_TILG2': 0.0823070625,
        'reseed_TILG1': 5.07893030502737,
        'reseed_TILV': 7668.0763846414,
        'reseed_CLV': 50.8840614011767,
        'reseed_CRES': 9.48358932320817,
        'reseed_CST': 0.006359695307559,
        'reseed_CSTUB': 0.000220595291471,

        # set from a mid point value not important for percistance, but important to stop inital high yeild!,
        # set to start of simulation start month(7) average,
        'LOG10CLVI': np.log10(50.8840614011767),
        'LOG10CRESI': np.log10(9.48358932320817),
        'LOG10CRTI': np.log10(122.557695833499),

    },
    ('oxford', 'store600'): {  # just an example of what is needed, should never be callable
        # site, mode specific parameters,
        'h2o_store_max_vol': 600,
        # reseed parameteres, set as mean of long term runs in june,
        'reseed_harv_delay': 20,

        'reseed_LAI': 1.80970469368663,
        'reseed_TILG2': 0,
        'reseed_TILG1': 2.99420228365586,
        'reseed_TILV': 7675.86610923919,
        'reseed_CLV': 45.5977252357025,
        'reseed_CRES': 8.79419880421888,
        'reseed_CST': 0,
        'reseed_CSTUB': 1.91975264109104E-11,

        # set from a mid point value not important for percistance, but important to stop inital high yeild!,
        # set to start of simulation start month(7) average,
        'LOG10CLVI': np.log10(45.5977252357025),
        'LOG10CRESI': np.log10(8.79419880421888),
        'LOG10CRTI': np.log10(155.186719177027),

    },

    ('eyrewell', 'store800'): {  # just an example of what is needed, should never be callable
        # site, mode specific parameters,
        'h2o_store_max_vol': 800,
        # reseed parameteres, set as mean of long term runs in june,
        'reseed_harv_delay': 20,

        'reseed_LAI': 1.89928116188838,
        'reseed_TILG2': 0,
        'reseed_TILG1': 5.08749730356038,
        'reseed_TILV': 7702.21068421423,
        'reseed_CLV': 50.9472259343913,
        'reseed_CRES': 9.49499427448814,
        'reseed_CST': 0,
        'reseed_CSTUB': 1.87884339242268E-11,

        # set from a mid point value not important for percistance, but important to stop inital high yeild!,
        # set to start of simulation start month(7) average,
        'LOG10CLVI': np.log10(50.9472259343913),
        'LOG10CRESI': np.log10(9.49499427448814),
        'LOG10CRTI': np.log10(120.887079046102),

    },

    ('oxford', 'store800'): {  # just an example of what is needed, should never be callable
        # site, mode specific parameters,
        'h2o_store_max_vol': 800,
        # reseed parameteres, set as mean of long term runs in june,
        'reseed_harv_delay': 20,

        'reseed_LAI': 1.8132001707278,
        'reseed_TILG2': 0,
        'reseed_TILG1': 3.05028976552842,
        'reseed_TILV': 7801.43360149441,
        'reseed_CLV': 45.6967500723288,
        'reseed_CRES': 8.81243862850446,
        'reseed_CST': 0,
        'reseed_CSTUB': 1.93662134671063E-11,

        # set from a mid point value not important for percistance, but important to stop inital high yeild!,
        # set to start of simulation start month(7) average,
        'LOG10CLVI': np.log10(45.6967500723288),
        'LOG10CRESI': np.log10(8.81243862850446),
        'LOG10CRTI': np.log10(152.833330523423),

    },
}

ibasals = {  # iterated a couple of times before landing on the mean july ibasals
    ('eyrewell', 'store400'): 0.7574,
    ('oxford', 'store400'): 0.7227,
    ('eyrewell', 'store600'): 0.7638,
    ('oxford', 'store600'): 0.7289,
    ('eyrewell', 'store800'): 0.7662,
    ('oxford', 'store800'): 0.7329,
}


def set_store_parameters(site, mode, params):
    params = copy.deepcopy(params)
    # site, scenario independent values
    # params['abs_max_irr']  set in basgra_parameter_sets.py
    params['use_storage'] = 1
    params['runoff_from_rain'] = 1
    params['calc_ind_store_demand'] = 0
    params['stor_full_refil_doy'] = 213
    params['irrigated_area'] = 1  # per hectare modelling
    params['I_h2o_store_vol'] = 1
    params['h2o_store_SA'] = 0
    params['runoff_area'] = 0
    params['runoff_frac'] = 0
    params['stor_leakage'] = 0
    params['stor_reserve_vol'] = 0

    params['stor_refill_losses'] = 0
    params['stor_irr_ineff'] = 0
    params['stor_refill_min'] = 0

    # site, mode specific parameters
    site_mode_spec_keys = ['h2o_store_max_vol', 'reseed_harv_delay', 'reseed_LAI', 'reseed_TILG2', 'reseed_TILG1',
                           'reseed_TILV', 'reseed_CLV', 'reseed_CRES', 'reseed_CST', 'reseed_CSTUB',
                           'LOG10CLVI', 'LOG10CRESI', 'LOG10CRTI']

    for k in site_mode_spec_keys:
        params[k] = site_mode_dep_params[(site, mode)][k]

    params['BASALI'] = ibasals[(site, mode)]
    return params


def get_store_reseed_trig_basal(site, mode):
    # percentages were middle of those defined for eyrewell/oxford irrigated
    reseed_trig = ibasals[(site, mode)] * 93.5 / 100
    reseed_basal = ibasals[(site, mode)] * 97.35 / 100
    return reseed_trig, reseed_basal


# todo this will have different seeds than the original???, perhaps just run on subset... of storylines... or re-run everything????, smaller sample...
# todo possibly best bet at this point is to only run the storylines that we need --> the hurt/scare/subset of 'most probable'
# todo re-running everything is not really an option time wise.

# todo run a test of every site/mode for a storyline
# todo perhaps run big suite over break
# todo scripts that need to be re-run with new storage systems:
# #### model runs ####
# Storylines/storyline_runs/historical_quantified_1yr_detrend.py # have run
# Storylines\storyline_runs\historical_quantified_1yr_trend.py # have run

# Storylines/storyline_runs/run_random_suite.py # have run
# Storylines/storyline_runs/run_nyr_suite.py # have run


# #### exports and plots ####
# Storylines/storyline_evaluation/plot_historical_trended.py # have run
# Storylines/storyline_evaluation/plot_historical_detrended.py # have run

# Storylines/storyline_evaluation/export_cum_percentile.py # have run
# Storylines\storyline_evaluation\plot_site_v_site.py # have run
# Storylines/storyline_evaluation/plots.py # have run
# Storylines/storyline_evaluation/storyline_slection/stories_to_ws_pg_threshold.py # todo running
# Storylines/storyline_evaluation/plot_cumulative_historical_v_modelled.py # run
# Storylines\final_plots\prob_and_pg_with_storage.py # have run

# todo review everything
if __name__ == '__main__':
    for k, v in site_mode_dep_params.items():
        print(k, v['h2o_store_max_vol'])
