"""
 Author: Matt Hanson
 Created: 17/02/2021 2:46 PM
 """
from Climate_Shocks.climate_shocks_env import storyline_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from BS_work.SWG.SWG_wrapper import *

if __name__ == '__main__':
    #todo run set up for new run styles and on dickie
    run_basgra = False

    if run_basgra:
        # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_key='0-base',
                           outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'pasture_growth_sims'),
                           nsims='all', padock_rest=True,
                           save_daily=True, description='initial baseline run')