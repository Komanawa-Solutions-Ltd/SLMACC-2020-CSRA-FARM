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
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from BS_work.SWG.SWG_wrapper import *

if __name__ == '__main__':
    #todo set up with new datsets run on dickie
    run_basgra = True

    if run_basgra:
    # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_key='0-base',
                           outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'pasture_growth_sims'),
                           nsims='all', padock_rest=True,
                           save_daily=True, description='initial baseline run note that this was run before fixing '
                                                        'the swg matching errors e.g. realisation cleaning')
