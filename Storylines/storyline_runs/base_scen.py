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
from BS_work.SWG.SWG_wrapper import *

if __name__ == '__main__':
    #todo set up with new datsets run on dickie , and REVIEW
    # todo may need to re-run with new irrigation restrictions...
    run_basgra = False  # to stop accidental re-run

    if run_basgra:
    # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_path=os.path.join(storyline_dir, '0-baseline.csv'),
                           outdir=os.path.join(default_pasture_growth_dir, 'baseline_sim'),
                           nsims=10000, padock_rest=True,
                           save_daily=True, description='initial baseline run after the realisation cleaning, '
                                                        'have not finalized irrigation restirctions',
                           verbose=True)
