"""
 Author: Matt Hanson
 Created: 16/02/2021 12:55 PM
 """

import itertools
import pandas as pd
import time
import ksl_env
import os
from Storylines.story_swg_iid_managment import storyline_swg_paths
from Climate_Shocks.climate_shocks_env import storyline_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from BS_work.SWG.SWG_wrapper import *

if __name__ == '__main__':
    # todo set up the check and deletion of any data that does not fit limits
    #todo run on dickie
    run_swg = True
    clean_data = False
    run_basgra = False

    # run swg
    print('running SWG')
    outdir = os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', '0-base')
    yml = os.path.join(outdir, '0-base.yml')
    if run_swg:
        create_yaml(outpath_yml=yml, outsim_dir=outdir,
                    nsims=10100,
                    storyline_path=os.path.join(storyline_dir, '0-baseline.csv'),
                    sim_name=None,
                    xlat=oxford_lat, xlon=oxford_lon)
        temp = run_SWG(yml, outdir, rm_npz=True, clean=False)
        print(temp)

    if clean_data: #todo make sure to save a copy to check cleaning process!
        print('cleaning_data')
        clean_swg(outdir, yml)

    if run_basgra:
    # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_key='0-base',
                           outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'pasture_growth_sims'),
                           nsims='all', padock_rest=True,
                           save_daily=True, description='initial baseline run')
