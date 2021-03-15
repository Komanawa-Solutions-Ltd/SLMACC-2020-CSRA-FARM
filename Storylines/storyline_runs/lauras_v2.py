"""
 Author: Matt Hanson
 Created: 15/03/2021 3:23 PM
 """

import pandas as pd
import ksl_env
import os
import itertools
import glob
from copy import deepcopy
from Storylines.storyline_building_support import base_events, map_irrigation, default_storyline_time
from Storylines.check_storyline import ensure_no_impossible_events
from Climate_Shocks import climate_shocks_env
from Pasture_Growth_Modelling.full_pgr_model_mp import default_pasture_growth_dir, run_full_model_mp, pgm_log_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from Pasture_Growth_Modelling.plot_full_model import plot_sims
from Pasture_Growth_Modelling.export_to_csvs import export_all_in_pattern

name = 'lauras_v2'
story_dir = os.path.join(climate_shocks_env.temp_storyline_dir, name)
if not os.path.exists(story_dir):
    os.makedirs(story_dir)

base_pg_outdir = os.path.join(default_pasture_growth_dir, name)
outputs_dir = os.path.join(climate_shocks_env.temp_output_dir, name)

for d in [story_dir, base_pg_outdir, outputs_dir]:
    if not os.path.exists(d):
        os.makedirs(d)


def make_storylines():
    test = pd.read_excel(r"M:\Shared drives\Z2003_SLMACC\storylines\blank_storylineWS120321.xlsx", skiprows=2,
                         index_col=[0, 1, 2, 3], header=None) #todo update path
    header_1 = test.iloc[0].ffill().values
    header_2 = test.iloc[1].values
    storylines = pd.DataFrame(index=pd.MultiIndex.from_arrays([default_storyline_time.year,
                                                               default_storyline_time.month], names=['year', 'month']),
                              columns=pd.MultiIndex.from_arrays([header_1, header_2]), data=test.iloc[2:].values)

    raise NotImplementedError()


def run_pasture_growth_mp():
    outdirs = [base_pg_outdir for e in os.listdir(story_dir)]
    paths = [os.path.join(story_dir, e) for e in os.listdir(story_dir)]
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


def export_and_plot_data():
    export_all_in_pattern(base_outdir=outputs_dir,
                          patterns=[
                              os.path.join(story_dir, '*.nc'),
                              os.path.join(os.path.dirname(story_dir), 'baseline_sim_no_pad', '*.nc')
                          ])
    for sm in ['eyrewell-irrigated', 'oxford-dryland', 'oxford-irrigated']:
        paths = glob.glob(os.path.join(story_dir, f'*{sm}.nc'))
        for p in paths:
            outdir = os.path.join(story_dir, 'plots', os.path.basename(p).replace('f-{sm}.nc', ''))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            data_paths = [p, f"D:/mh_unbacked/SLMACC_2020/pasture_growth_sims/baseline_sim_no_pad/0-baseline-{sm}.nc"]

            plot_sims(data_paths,
                      plot_ind=False, nindv=100, save_dir=outdir, show=False, figsize=(20, 20),
                      daily=False, ex_save=f'{sm}-')


if __name__ == '__main__':
    run = False
    plot_export = False
    export_csv = True
    make_storylines()
    if run:
        run_pasture_growth_mp()
    if plot_export:
        export_and_plot_data()
