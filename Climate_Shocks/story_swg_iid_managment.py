"""
 Author: Matt Hanson
 Created: 11/02/2021 12:32 PM
 """
import ksl_env
import os

# todo something to keep track of the locations of all of the data

storyline_swg_paths = {
    # key: (storyline_path(csv file),
    #       swg_path(direictory for SWG files),
    #       nsims_aval(number of sims in the SWG,
    #       simlen(number of days in the simulation)

    'test100': (  # used to test and profile run...
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BS_work/SWG/v7.csv'),  # storyline_path(csv file)
        os.path.join(ksl_env.slmmac_dir_unbacked, '100_rm_npz'),  # swg_path(direictory for SWG files),
        100,  # nsims_aval(number of sims in the SWG,
        730  # simlen(number of days in the simulation)
    ),

    'test1000': (  # used to test and profile run...
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BS_work/SWG/v7.csv'),  # storyline_path(csv file)
        os.path.join(ksl_env.slmmac_dir_unbacked, '1000_rm_npz'),  # swg_path(direictory for SWG files),
        1000,  # nsims_aval(number of sims in the SWG,
        730  # simlen(number of days in the simulation)
    ),

    'test10000': (  # used to test and profile run...
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BS_work/SWG/v7.csv'),  # storyline_path(csv file)
        os.path.join(ksl_env.slmmac_dir_unbacked, '10000_rm_npz'),  # swg_path(direictory for SWG files),
        10000,  # nsims_aval(number of sims in the SWG,
        730  # simlen(number of days in the simulation)
    ),

}
