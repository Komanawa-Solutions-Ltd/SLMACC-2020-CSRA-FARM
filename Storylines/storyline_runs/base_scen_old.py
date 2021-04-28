"""
 Author: Matt Hanson
 Created: 16/02/2021 12:55 PM
 """

import itertools
import pandas as pd
import time
import ksl_env
import os
from Climate_Shocks.climate_shocks_env import storyline_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth, default_pasture_growth_dir
from Pasture_Growth_Modelling.plot_full_model import plot_sims
from BS_work.SWG.SWG_wrapper import *
from Pasture_Growth_Modelling.export_to_csvs import export_all_in_pattern
from Storylines.storyline_evaluation.storyline_eval_support import extract_additional_sims
import warnings
warnings.warn('this is the old baseline, it is depreciated!!!!')

if __name__ == '__main__':
    run_basgra = True  # to stop accidental re-run
    plot_results = False
    export = True
    prob_pg = True
    if run_basgra:
        # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_path=os.path.join(storyline_dir, '0-baseline_1yr.csv'),
                           outdir=os.path.join(default_pasture_growth_dir, 'baseline_sim_no_pad'),
                           nsims=10000, padock_rest=False,
                           save_daily=True, description='initial baseline run after the realisation cleaning',
                           verbose=True)

    if plot_results:
        path_list = [
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim_no_pad\0-baseline-eyrewell-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim_no_pad\0-baseline-oxford-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim_no_pad\0-baseline-oxford-dryland.nc",

        ]
        for p in path_list:
            plot_sims(data_paths=p, plot_ind=True, nindv=50, save_dir=None, show=True, figsize=(11, 8),
                      daily=False,
                      )

    if export:
        export_all_in_pattern(base_outdir=os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'Baseline'),
                              patterns=os.path.join(ksl_env.slmmac_dir_unbacked,
                                                    "pasture_growth_sims/baseline_sim_no_pad/*.nc"),
                              )

    if prob_pg:
        data = extract_additional_sims(storyline_dir,
                                       os.path.join(default_pasture_growth_dir, 'baseline_sim_no_pad'), 3)
        data.to_csv(os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'Baseline','prob_iid_pg.csv'))
