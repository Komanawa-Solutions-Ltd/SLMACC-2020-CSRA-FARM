"""
 Author: Matt Hanson
 Created: 15/02/2021 2:36 PM
 """
import time
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
import ksl_env
import os
import pandas as pd
import itertools
from Climate_Shocks.climate_shocks_env import storyline_dir

def check_full_model_run_time():
    out = pd.DataFrame(index=range(5), columns=['nsims', 'time', 'paddock_res', 'save_daily'])
    t = time.time()

    run_pasture_growth(storyline_path=os.path.join(storyline_dir,'0-baseline.csv'),
                       outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'test_full_modelbase_10000'),
                       nsims=10000, padock_rest=False, mode_sites=[('irrigated', 'eyrewell')],
                       save_daily=True, description='to test functinality')

    out.loc[0, 'time'] = time.time() - t
    out.loc[0, 'nsims'] = 10000
    out.loc[0, 'paddock_res'] = False
    out.loc[0, 'save_daily'] = True


    for i, (num, pad, daily) in enumerate(itertools.product([100, 1000], [True, False], [True, False])):
        t = time.time()
        if pad:
            val = 'paddock'
        else:
            val = 'simple'
        if daily:
            dval = 'daily'
        else:
            dval = 'monthly'
        run_pasture_growth(storyline_path=os.path.join(storyline_dir,'0-baseline.csv'),
                           outdir=os.path.join(ksl_env.slmmac_dir_unbacked,
                                               'test_full_model0-base{}_{}_{}'.format(num, val, dval)),
                           nsims=num, padock_rest=pad, mode_sites=[('irrigated', 'eyrewell')],
                           save_daily=daily, description='to test functinality')

        out.loc[i + 1, 'time'] = time.time() - t
        out.loc[i + 1, 'nsims'] = num
        out.loc[i + 1, 'paddock_res'] = pad
        out.loc[i + 1, 'save_daily'] = daily

    out.to_csv(os.path.join(ksl_env.slmmac_dir, 'time_test_basgra_.csv'))

if __name__ == '__main__':
    check_full_model_run_time()