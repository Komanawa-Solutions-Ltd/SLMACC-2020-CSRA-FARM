"""
 Author: Matt Hanson
 Created: 11/02/2021 12:32 PM
 """
import ksl_env
import os
from Climate_Shocks import climate_shocks_env

# something to keep track of the locations of all of the data

storyline_swg_paths = {
    # key: (storyline_path(csv file),
    #       swg_path(direictory for SWG files),
    #       nsims_aval(number of sims in the SWG,
    #       simlen(number of days in the simulation)

    'test100': (  # used to test and profile run...
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BS_work/SWG/v7.csv'),  # storyline_path(csv file)
        os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', '100_rm_npz'),  # swg_path(direictory for SWG files),
        100,  # nsims_aval(number of sims in the SWG,
        730  # simlen(number of days in the simulation)
    ),

    'test1000': (  # used to test and profile run...
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BS_work/SWG/v7.csv'),  # storyline_path(csv file)
        os.path.join(ksl_env.slmmac_dir_unbacked,  'SWG_runs','1000_rm_npz'),  # swg_path(direictory for SWG files),
        1000,  # nsims_aval(number of sims in the SWG,
        730  # simlen(number of days in the simulation)
    ),

    'test10000': (  # used to test and profile run...
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BS_work/SWG/v7.csv'),  # storyline_path(csv file)
        os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', '10000_rm_npz'),  # swg_path(direictory for SWG files),
        10000,  # nsims_aval(number of sims in the SWG,
        730  # simlen(number of days in the simulation)
    ),

    '0-base': (  # baseline option as of 2021-02-16
        os.path.join(climate_shocks_env.storyline_dir, '0-baseline.csv'),  # storyline_path(csv file)
        os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', '0-base'),  # swg_path(direictory for SWG files),
        10000,  # nsims_aval(number of sims in the SWG, #todo update once data is cleaned
        365 * 3  # simlen(number of days in the simulation)
    ),

}

# todo need to run something here to capture the data as or once it is generated.