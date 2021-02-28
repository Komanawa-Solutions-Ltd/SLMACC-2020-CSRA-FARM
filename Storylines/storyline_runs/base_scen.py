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

if __name__ == '__main__':
    # todo re-run with new restriction data
    run_basgra = False  # to stop accidental re-run
    plot_results = True

    if run_basgra:
        # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_path=os.path.join(storyline_dir, '0-baseline.csv'),
                           outdir=os.path.join(default_pasture_growth_dir, 'baseline_sim'),
                           nsims=10000, padock_rest=True,
                           save_daily=True, description='initial baseline run after the realisation cleaning, '
                                                        'have not finalized irrigation restirctions',
                           verbose=True)

    if plot_results:
        path_list = [
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-paddock-mean-eyrewell-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-paddock-mean-oxford-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-eyrewell-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-oxford-irrigated.nc",
        ]
        path_list = [
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-paddock_2-eyrewell-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-paddock_3-eyrewell-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-paddock_0-eyrewell-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-paddock_1-eyrewell-irrigated.nc",
        ]

        plot_sims(data_paths=path_list, plot_ind=True, nindv=30, save_dir=None, show=True, figsize=(11, 8),
                  daily=False
                  )
