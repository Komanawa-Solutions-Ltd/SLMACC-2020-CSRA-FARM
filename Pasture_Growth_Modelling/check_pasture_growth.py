"""
 Author: Matt Hanson
 Created: 15/02/2021 10:34 AM
 """
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
import ksl_env
import os

if __name__ == '__main__':
    # todo plot and compair multiple paddock vs non paddock!
    # todo test against storyline with non zero restriction data
    run_pasture_growth(storyline_key='test100', outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'test_full_model'),
                       nsims='all', padock_rest=True, mode_sites=[('irrigated', 'eyrewell'),
                                                                  ('irrigated', 'oxford'), ],
                       save_daily=True, description='to test functinality')
