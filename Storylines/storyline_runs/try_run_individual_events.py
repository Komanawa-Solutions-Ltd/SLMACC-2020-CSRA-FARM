"""
 Author: Matt Hanson
 Created: 19/02/2021 2:49 PM
 """
import os
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from BS_work.SWG.SWG_wrapper import *
from Storylines.storyline_building_support import make_sampling_options, map_irrigation, prev_month, base_events
from BS_work.SWG.SWG_multiprocessing import run_swg_mp

# historical for running and checking the ability to create individual events, see also BS_work\SWG\check_1_month_runs.py

individual_dir = os.path.join(temp_storyline_dir, 'individual_runs')


def make_storyline_files():
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)
    # there are 69 unique event/month comparisons.

    all_events = make_sampling_options()  # this is just to exaime swg data,


    for m in range(1, 13):
        for e in all_events[m]:
            fname = 'm{:02d}-{}.csv'.format(m, '-'.join(e))

            pm = prev_month[m]
            ppm = prev_month[prev_month[m]]
            py = 2026
            if m == 1:
                py += -1
            ppy = 2026
            if m == 1 or m == 2:
                ppy += -1
            with open(os.path.join(individual_dir, fname), 'w') as f:
                f.write('month,year,temp_class,precip_class,rest\n')
                f.write('{},{},{},{},{}\n'.format(m, 2026, *e[0:-1],
                                                  map_irrigation(m, e[-1])))  # really map restrictions


if __name__ == '__main__':
    make_files = True
    make_weather = True
    run_basgra = False
    detrended_vcf =r"D:\SLMMAC_SWG_detrend_test_nPrecip\detrend_event_data_new_p_fixed.csv"

    n = 1000

    if make_files:
        make_storyline_files()

    # just thrown togeather so I can look at produced data
    # run swg
    if make_weather:
        print('running SWG')

        storylines = []
        outdirs = []
        for p in os.listdir(individual_dir):
            storylines.append(os.path.join(individual_dir, p))
            outdir = os.path.join(ksl_env.unbacked_dir, 'SWG_runs', 'try_individual_nPrecip', p.split('.')[0])
            outdirs.append(outdir)
        run_swg_mp(storyline_paths=storylines, outdirs=outdirs, ns=n, base_dirs=r"D:\SLMMAC_SWG_detrend_test_nPrecip",
                   vcfs=detrended_vcf,
                   log_path=r"D:\mh_unbacked\SLMACC_2020\SWG_runs\logs\try_individual_newp_02-22-16_35.csv",
                   pool_size=1)
