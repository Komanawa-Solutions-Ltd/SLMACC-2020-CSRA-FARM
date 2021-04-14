"""
 Author: Matt Hanson
 Created: 25/02/2021 2:08 PM
 """

import pandas as pd
import ksl_env
import os
import itertools
import glob
from copy import deepcopy
from Storylines.storyline_building_support import base_events, map_irrigation
from Storylines.check_storyline import ensure_no_impossible_events
from Climate_Shocks import climate_shocks_env
from Pasture_Growth_Modelling.full_pgr_model_mp import default_pasture_growth_dir, run_full_model_mp, pgm_log_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from Pasture_Growth_Modelling.plot_full_model import plot_sims
from Pasture_Growth_Modelling.export_to_csvs import export_all_in_pattern

default_lauras_story_dir = os.path.join(climate_shocks_env.temp_storyline_dir, 'lauras_run')
if not os.path.exists(default_lauras_story_dir):
    os.makedirs(default_lauras_story_dir)

inital_laura_dir = os.path.join(default_pasture_growth_dir, 'lauras')


def make_storylines(rest_quantiles=[0.75, 0.95], no_irr_event=0.5):
    years = [1, 2, 3]
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasons_m = [(6, 7, 8), (9, 10, 11), (12, 1, 2), (3, 4, 5)]

    # 'Cold', 'Drought', 'Restriction', 'Hot'

    inputdata = pd.read_excel(os.path.join(ksl_env.slmmac_dir, 'storylines\StorylineSpreadsheetWSLB.xlsx'),
                              header=[0, 1, 2], index_col=0).transpose()
    inputdata.columns = inputdata.columns.str.strip()
    for k in inputdata.keys():
        print(k)
        data = pd.DataFrame(index=pd.date_range('2024-07-01', '2027-06-01', freq='MS'),
                            columns=['precip_class', 'temp_class', 'rest'])
        data.index.name = 'date'
        data.loc[:, 'year'] = data.index.year
        data.loc[:, 'month'] = data.index.month
        data.loc[:, 'rest'] = 0
        for i, y, m in data.loc[:, ['year', 'month']].itertuples(True, None):
            t, p, r, rp = base_events[m]
            data.loc[i, 'precip_class'] = p
            data.loc[i, 'temp_class'] = t

        temp = inputdata.loc[:, k]

        for y, (s, sms) in itertools.product(years, zip(seasons, seasons_m)):
            svs = [int(e) for e in temp.loc[y, s, 'Cold'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y

                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'temp_class'] = 'C'

            svs = [int(e) for e in temp.loc[y, s, 'Hot'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue
                    data.loc[idx, 'temp_class'] = 'H'

            svs = [int(e) for e in temp.loc[y, s, 'Drought'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'precip_class'] = 'D'

            svs = [int(e) for e in temp.loc[y, s, 'Wet'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'precip_class'] = 'W'

            svs = [int(e) for e in temp.loc[y, s, 'Restriction'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'rest'] = 1
        pass
        try:
            ensure_no_impossible_events(data)
        except ValueError as v:
            print(f'{k}: \n {v}')

        data.loc[:, 'precip_class_prev'] = data.loc[:, 'precip_class'].shift(1).fillna('A')
        for q in rest_quantiles:
            data_out = deepcopy(data)
            data_out.loc[:, 'rest_code'] = data.loc[:, 'rest']
            q1 = q - no_irr_event
            for i in data.index:
                data_out.loc[i, 'rest'] = map_irrigation(m=data_out.loc[i, 'month'],
                                                         rest_quantile=q1 * data_out.loc[i, 'rest'] + no_irr_event,
                                                         precip=data_out.loc[i, 'precip_class'],
                                                         prev_precip=data_out.loc[i, 'precip_class_prev'])

            data_out.to_csv(os.path.join(default_lauras_story_dir,
                                         f'{k}-rest-{int(no_irr_event * 100)}-{int(q * 100)}.csv'))


def run_pasture_growth_mp():
    base_outdir = os.path.join(default_pasture_growth_dir, 'lauras')
    if not os.path.exists(base_outdir):
        os.makedirs(base_outdir)

    outdirs = [base_outdir for e in os.listdir(default_lauras_story_dir)]
    paths = [os.path.join(default_lauras_story_dir, e) for e in os.listdir(default_lauras_story_dir)]
    run_full_model_mp(
        storyline_path_mult=paths,
        outdir_mult=outdirs,
        nsims_mult=10000,
        log_path=os.path.join(pgm_log_dir, 'lauras'),
        description_mult='a first run of Lauras storylines',
        padock_rest_mult=False,
        save_daily_mult=True,
        verbose=False

    )


def run_pasture_growth_normal():
    base_outdir = os.path.join(default_pasture_growth_dir, 'lauras')
    if not os.path.exists(base_outdir):
        os.makedirs(base_outdir)

    nsims = 1
    paths = [os.path.join(default_lauras_story_dir, e) for e in os.listdir(default_lauras_story_dir)]
    outdirs = [base_outdir for e in paths]

    for p, od in zip(paths, outdirs):
        print(p)
        run_pasture_growth(storyline_path=p, outdir=od, nsims=nsims, padock_rest=False,
                           save_daily=True, description='', verbose=True,
                           n_parallel=1)


if __name__ == '__main__':
    run = False
    run_missing = False
    plot = False
    export_csv = True
    if run:
        run_pasture_growth_mp()
    if run_missing:
        inital_laura_dir = os.path.join(default_pasture_growth_dir, 'lauras')
        sp = r'D:/mh_unbacked\SLMACC_2020\temp_storyline_files\lauras_run\Event Every Season Y1 and Y3-rest-50-95.csv'
        run_pasture_growth(storyline_path=sp, outdir=inital_laura_dir, nsims=10000, padock_rest=False,
                           save_daily=True, description='', verbose=True,
                           n_parallel=1)

    if plot:
        data_paths = []

        inital_laura_dir = os.path.join(default_pasture_growth_dir, 'lauras')
        for sm in ['eyrewell-irrigated', 'oxford-dryland', 'oxford-irrigated']:
            data_paths = glob.glob(os.path.join(inital_laura_dir, f'*{sm}.nc')) + [
                f"D:/mh_unbacked/SLMACC_2020/pasture_growth_sims/baseline_sim_no_pad/0-baseline-{sm}.nc"]
            outdir = os.path.join(inital_laura_dir, 'plots', sm)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            plot_sims(data_paths,
                      plot_ind=False, nindv=100, save_dir=outdir, show=False, figsize=(20, 20),
                      daily=False)
    if export_csv:
        export_all_in_pattern(base_outdir=os.path.join(ksl_env.slmmac_dir, 'output_pgr', 'inital_laura_runs'),
                              patterns=[
                                  os.path.join(inital_laura_dir, '*.nc'),
                                  os.path.join(os.path.dirname(inital_laura_dir), 'baseline_sim_no_pad', '*.nc')
                              ])
