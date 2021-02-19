"""
 Author: Matt Hanson
 Created: 19/02/2021 2:49 PM
 """
import os
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from BS_work.SWG.SWG_wrapper import *
from Storylines.storyline_building_support import make_sampling_options, map_irrigation, prev_month, base_events

# todo run a bunch of storylines for each individual event to see impact of just that event. # proceed with
#  2 normal months, run 2500 (rethink based on number of individaul events
#  samples for each one this suggests less than a 1 kg error on mean monthly pasture growth

def make_storyline_files():
    outdir = os.path.join(temp_storyline_dir, 'individual_runs')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # there are 69 unique event/month comparisons.

    all_events = make_sampling_options()

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
            map_irrigation(1,1)
            with open(os.path.join(outdir, fname), 'w') as f:
                f.write('month,year,precip_class,temp_class,rest\n')
                f.write('{},{},{},{},{}\n'.format(ppm, ppy, *base_events[ppm][0:-1],
                                                  map_irrigation(ppm, base_events[ppm][-1])))
                f.write('{},{},{},{},{}\n'.format(pm, py, *base_events[pm][0:-1],
                                                  map_irrigation(pm, base_events[pm][-1])))
                f.write('{},{},{},{},{}\n'.format(m, 2026, *e[0:-1],
                                                  map_irrigation(m, e[-1])))  # todo really map restrictions


if __name__ == '__main__':
    make_files = True
    make_weather = False
    run_basgra = False
    n = 500 #todo consider carefully

    if make_files:
        make_storyline_files()

    # todo below this had not been managed, also consider multiprocessing here !
    # run swg
    print('running SWG')

    outdir = os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', '0-base')
    yml = os.path.join(outdir, '0-base.yml')
    if make_weather:

        create_yaml(outpath_yml=yml, outsim_dir=outdir,
                    nsims=n,
                    storyline_path=os.path.join(storyline_dir, '0-baseline.csv'),
                    sim_name=None,
                    xlat=oxford_lat, xlon=oxford_lon)
        temp = run_SWG(yml, outdir, rm_npz=True, clean=False)
        print(temp)

    if run_basgra:
    # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_key='0-base',
                           outdir=os.path.join(ksl_env.slmmac_dir_unbacked, 'pasture_growth_sims'),
                           nsims='all', padock_rest=True,
                           save_daily=True, description='initial baseline run note that this was run before fixing '
                                                        'the swg matching errors e.g. realisation cleaning')

