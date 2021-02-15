"""
 Author: Matt Hanson
 Created: 15/02/2021 10:34 AM
 """
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
import ksl_env
import os
#from memory_profiler import profile


#@profile
def test_mem():
    # todo plot and compair multiple paddock vs non paddock!
    # todo test against storyline with non zero restriction data
    run_pasture_growth(storyline_key='test100', outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'test_full_model'),
                       nsims='all', padock_rest=True, mode_sites=[('irrigated', 'eyrewell'),
                                                                  ('irrigated', 'oxford'), ],
                       save_daily=False, description='to test functinality')


# outputs of mem profile:
# Line #    Mem usage    Increment  Occurences   Line Contents
# ============================================================
#     10    118.6 MiB    118.6 MiB           1   @profile
#     11                                         def test_mem():
#     12                                             # plot and compair multiple paddock vs non paddock!
#     13                                             # test against storyline with non zero restriction data
#     14    118.6 MiB      0.0 MiB           1       run_pasture_growth(storyline_key='test100', outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'test_full_model'),
#     15    118.6 MiB      0.0 MiB           1                          nsims='all', padock_rest=True, mode_sites=[('irrigated', 'eyrewell'),
#     16    118.6 MiB      0.0 MiB           1                                                                     ('irrigated', 'oxford'), ],
#     17    140.3 MiB     21.7 MiB           1                          save_daily=False, description='to test functinality')


if __name__ == '__main__':
    test_mem()
