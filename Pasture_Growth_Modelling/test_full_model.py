"""
 Author: Matt Hanson
 Created: 26/02/2021 8:57 AM
 """
# todo get this running!

from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth, default_pasture_growth_dir
from Climate_Shocks import climate_shocks_env
import os

if __name__ == '__main__':
    run_pasture_growth(
        storyline_path=os.path.join(climate_shocks_env.storyline_dir, '0-baseline.csv'),
        outdir=os.path.join(default_pasture_growth_dir, 'test_pg_ex_swg'), nsims=100, padock_rest=True,
        save_daily=True, description='', verbose=False)
