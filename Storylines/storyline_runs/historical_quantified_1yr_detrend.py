"""
 Author: Matt Hanson
 Created: 15/03/2021 3:23 PM
 """

import pandas as pd
import numpy as np
import ksl_env
import os
import itertools
import glob
from copy import deepcopy
from Climate_Shocks import climate_shocks_env
from Storylines.storyline_building_support import base_events, map_storyline_rest, default_storyline_time, \
    default_mode_sites, map_irr_quantile_from_rest, month_fchange, month_len
from Storylines.check_storyline import ensure_no_impossible_events
from Storylines.storyline_evaluation.storyline_eval_support import extract_additional_sims
from Climate_Shocks import climate_shocks_env
from Pasture_Growth_Modelling.full_pgr_model_mp import default_pasture_growth_dir, run_full_model_mp, pgm_log_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from Pasture_Growth_Modelling.plot_full_model import plot_sims
from Pasture_Growth_Modelling.export_to_csvs import export_all_in_pattern
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_smd_monthly
from BS_work.SWG.SWG_wrapper import get_monthly_smd_mean_detrended

name = 'historical_quantified_1yr_detrend'
story_dir = os.path.join(climate_shocks_env.temp_storyline_dir, name)
if not os.path.exists(story_dir):
    os.makedirs(story_dir)

base_pg_outdir = os.path.join(default_pasture_growth_dir, name)
outputs_dir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', name)

for d in [story_dir, base_pg_outdir, outputs_dir]:
    if not os.path.exists(d):
        os.makedirs(d)


def make_storylines():
    data = get_vcsn_record('detrended2')
    data.loc[:, 'day'] = data.index.day
    data.loc[:, 'month'] = data.index.month
    data.loc[:, 'year'] = data.index.year
    data = data.loc[~((data.month == 2) & (data.day == 29))]

    rest = get_restriction_record('detrended2')
    rest.loc[:, 'day'] = rest.index.day
    rest.loc[:, 'month'] = rest.index.month
    rest.loc[:, 'year'] = rest.index.year
    rest = rest.loc[~((rest.month == 2) & (rest.day == 29))]
    rest.loc[:, 'f_rest'] = [rc / month_len[m] for rc, m in
                             rest.loc[:, ['f_rest', 'month']].itertuples(False, None)]
    rest = rest.groupby(['year', 'month']).sum()

    # calc SMA
    data.loc[:, 'sma'] = calc_smd_monthly(data.rain, data.pet, data.index) - data.loc[:, 'doy'].replace(
        get_monthly_smd_mean_detrended(leap=False))

    data.loc[:, 'wet'] = data.loc[:, 'rain'] >= 0.1
    data.loc[:, 'dry'] = data.loc[:, 'sma'] <= -15
    data.loc[:, 'hot'] = data.loc[:, 'tmax'] >= 25
    data.loc[:, 'cold'] = ((data.loc[:, 'tmin'] +
                            data.loc[:, 'tmax']) / 2).rolling(3).mean().fillna(method='bfill') <= 7
    data = data.groupby(['year', 'month']).sum()

    ndays_wet = {  # todo definition hard coded in
        # todo CHange to NEW EVENTS!
        'org': {  # this is the best value!
            5: 14,
            6: 11,
            7: 11,
            8: 13,
            9: 13,
        }
    }
    for v in ndays_wet.values():
        v.update({
            1: 99,
            2: 99,
            3: 99,
            4: 99,
            10: 99,
            11: 99,
            12: 99,
        })

    for y in range(1972, 2019):
        t = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ]) + y
        tm = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, ]
        idx = list(zip(t, tm))
        temp = pd.DataFrame(index=np.arange(12),
                            columns=['year', 'month', 'temp_class', 'precip_class', 'rest', 'rest_per']
                            )
        temp.loc[:, 'year'] = [2024, 2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025, 2025, 2025, 2025, ]
        temp.loc[:, 'month'] = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, ]
        # todo events hard coded in
        # todo CHange to NEW EVENTS!
        temp.loc[:, 'rest'] = rest.loc[idx, 'f_rest'].round(2).values

        temp.loc[:, 'temp_class'] = 'A'
        idx2 = data.loc[idx, 'hot'] >= 7
        temp.loc[idx2.values, 'temp_class'] = 'H'
        idx2 = data.loc[idx, 'cold'] >= 10
        temp.loc[idx2.values, 'temp_class'] = 'C'

        temp.loc[:, 'precip_class'] = 'A'
        idx2 = data.loc[idx, 'dry'] >= 10
        temp.loc[idx2.values, 'precip_class'] = 'D'
        temp.loc[np.in1d(temp.month, [6, 7, 8]), 'precip_class'] = 'A'
        idx2 = data.loc[idx, 'wet'] >= [ndays_wet['org'][m] for m in temp.loc[:, 'month']]
        temp.loc[idx2.values, 'precip_class'] = 'W'

        temp.loc[:, 'precip_class_prev'] = temp.loc[:, 'precip_class'].shift(1).fillna('A')
        temp.loc[:, 'rest_per'] = [
            map_irr_quantile_from_rest(m=m,
                                       rest_val=rq,
                                       precip=p,
                                       prev_precip=pp) for m, rq, p, pp in temp.loc[:,
                                                                           ['month', 'rest',
                                                                            'precip_class',
                                                                            'precip_class_prev'
                                                                            ]].itertuples(False, None)]
        temp.to_csv(os.path.join(story_dir, f'sl-{y}.csv'))


def run_pasture_growth_mp(re_run):
    outdirs = [base_pg_outdir for e in os.listdir(story_dir)]
    paths = [os.path.join(story_dir, e) for e in os.listdir(story_dir)]
    run_full_model_mp(
        storyline_path_mult=paths,
        outdir_mult=outdirs,
        nsims_mult=1000,
        log_path=os.path.join(pgm_log_dir, 'lauras_1yr'),
        description_mult='a first run of Lauras storylines',
        padock_rest_mult=False,
        save_daily_mult=True,
        verbose=False,
        re_run=re_run

    )


def export_and_plot_data():
    export_all_in_pattern(base_outdir=outputs_dir,
                          patterns=[
                              os.path.join(base_pg_outdir, '*.nc'),
                          ])
    for sm in ['eyrewell-irrigated', 'oxford-dryland', 'oxford-irrigated']:
        site, mode = sm.split('-')
        data = get_historical_1yr_pg_prob(sm.split('-')[0], sm.split('-')[1])
        data.to_csv(os.path.join(outputs_dir, f'IID_probs_pg.csv'))
        paths = glob.glob(os.path.join(base_pg_outdir, f'*{sm}.nc'))
        for p in paths:
            outdir = os.path.join(outputs_dir, sm, 'plots')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            data_paths = [p]

            plot_sims(data_paths,
                      plot_ind=False, nindv=100, save_dir=outdir, show=False, figsize=(20, 20),
                      daily=False, ex_save=os.path.basename(p).replace('.nc', ''), site=site, mode=mode)


def get_historical_1yr_pg_prob(site, mode):
    data = extract_additional_sims(story_dir, base_pg_outdir, 1)

    rename_dict = {f'{site}-{mode}_pg': 'pgr', f'{site}-{mode}_pgra': 'pgra', f'log10_prob_{mode}': 'prob'}

    data.loc[:, 'plotlabel'] = [idv[0:12] for i, idv in data.loc[:, ['ID']].itertuples(True, None)]
    data = data.rename(columns=rename_dict)
    return data


if __name__ == '__main__':
    # todo re-run with new event data
    re_run = False
    make_st = True
    run = True
    plot_export = True
    pg_prob = True
    if make_st:
        make_storylines()
    if run:
        run_pasture_growth_mp(re_run)
    if plot_export:
        export_and_plot_data()
    if pg_prob:
        for mode, site in default_mode_sites:
            get_historical_1yr_pg_prob(site=site, mode=mode)
