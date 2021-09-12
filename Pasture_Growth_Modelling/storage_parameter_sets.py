"""
 Author: Matt Hanson
 Created: 10/09/2021 10:01 AM
 """
import copy

import numpy as np

# todo fill out and add into the thing:
#  data at: https://docs.google.com/spreadsheets/d/1-_ZU9wCvGYjn1j2Z_m9ybtziW_fLWHnfFqLd7CkPN5s/edit?usp=sharing


site_mode_dep_params = {  # todo need to manually mange these
    ('ex_site', 'ex_mode'): {  # todo just an example of what is needed, should never be callable
        # site, mode specific parameters,
        'h2o_store_max_vol': 10000,

        # reseed parameteres, set as mean of long term runs in june,
        'reseed_harv_delay': 20,
        'reseed_LAI': 1.840256e+00,
        'reseed_TILG2': 2.194855e+00,
        'reseed_TILG1': 4.574009e+00,
        'reseed_TILV': 6.949611e+03,
        'reseed_CLV': 5.226492e+01,
        'reseed_CRES': 9.727732e+00,
        'reseed_CST': 1.677470e-01,
        'reseed_CSTUB': 0,
        'BASALI': 0.747,

        # set from a mid point value not important for percistance, but important to stop inital high yeild!,
        # set to start of simulation start month(7) average,
        'LOG10CLVI': np.log10(51.998000),
        'LOG10CRESI': np.log10(9.627059),
        'LOG10CRTI': np.log10(125.966156),

        # days harvest matrix keys
        'reseed_trig': 0.696,
        'reseed_basal': 0.727,
    }

}


def set_store_parameters(site, mode, params):  # todo confirm with WS
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

    params['stor_refill_losses'] = 0  # todo not sure whether this will be independently set
    params['stor_irr_ineff'] = 0  # todo not sure whether this will be independently set
    params['stor_refill_min'] = 0  # todo not sure whether this will be independently set

    # site, mode specific parameters
    site_mode_spec_keys = ['h2o_store_max_vol', 'reseed_harv_delay', 'reseed_LAI', 'reseed_TILG2', 'reseed_TILG1',
                           'reseed_TILV', 'reseed_CLV', 'reseed_CRES', 'reseed_CST', 'reseed_CSTUB', 'BASALI',
                           'LOG10CLVI', 'LOG10CRESI', 'LOG10CRTI']

    for k in site_mode_spec_keys:
        params[k] = site_mode_dep_params[(site, mode)][k]

    return params


def get_store_reseed_trig_basal(site, mode):
    return site_mode_dep_params[(site, mode)]['reseed_trig'], site_mode_dep_params[(site, mode)]['reseed_basal']

# todo scripts that need to be re-run with new storage systems:
# #### model runs ####
# Storylines/storyline_runs/historical_quantified_1yr_detrend.py
# Storylines\storyline_runs\historical_quantified_1yr_trend.py
# Storylines/storyline_runs/run_random_suite.py
# Storylines/storyline_runs/run_nyr_suite.py


# #### exports and plots ####
# Storylines/storyline_evaluation/plot_historical_detrended.py
# Storylines/storyline_evaluation/export_cum_percentile.py
# Storylines/storyline_evaluation/plot_cumulative_historical_v_modelled.py
# Storylines/storyline_evaluation/plot_historical_trended.py
# Storylines\storyline_evaluation\plot_site_v_site.py
# Storylines/storyline_evaluation/plots.py
# Storylines/storyline_evaluation/storyline_slection/stories_to_ws_pg_threshold.py
