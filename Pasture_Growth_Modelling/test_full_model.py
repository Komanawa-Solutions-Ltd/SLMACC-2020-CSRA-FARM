"""
 Author: Matt Hanson
 Created: 26/02/2021 8:57 AM
 """

from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from Climate_Shocks import climate_shocks_env
import os

if __name__ == '__main__':
    run_pasture_growth(
        storyline_path=os.path.join(climate_shocks_env.storyline_dir, '0-baseline.csv'),
        outdir=r"C:\Users\dumon\Downloads\test_pg_ex_swg", nsims=100, padock_rest=True,
        save_daily=True, description='', verbose=False)
    # todo make sure new reseeded works appropriately