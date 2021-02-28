"""
 the goal is to generate the full suite of options to pick from.
 Author: Matt Hanson
 Created: 24/02/2021 10:48 AM
 """
import os
import datetime
import shutil
import ksl_env
import pandas as pd
from Climate_Shocks import climate_shocks_env
from BS_work.SWG.SWG_wrapper import default_vcf, default_base_dir, clean_swg
from Storylines.storyline_building_support import make_sampling_options
from BS_work.SWG.SWG_multiprocessing import run_swg_mp

individual_dir = os.path.join(climate_shocks_env.temp_storyline_dir, 'individual_runs')
log_dir = r"D:\mh_unbacked\SLMACC_2020\SWG_runs\logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def make_storyline_files():
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)
    # there are 69 unique event/month comparisons.

    all_events = make_sampling_options()

    for m in range(1, 13):
        for e in all_events[m]:
            fname = 'm{:02d}-{}.csv'.format(m, '-'.join(e))

            with open(os.path.join(individual_dir, fname), 'w') as f:
                f.write('month,year,temp_class,precip_class,rest\n')
                f.write('{},{},{},{},{}\n'.format(m, 2025, *e[0:-1], 0))


def generate_SWG_output_support(vcfs=default_vcf, base_dirs=default_base_dir):
    make_storyline_files()
    # delete the old outputs
    shutil.rmtree(os.path.join(default_base_dir, 'Output'))

    print('running SWG to create the output files')
    storylines = []
    outdirs = []
    for p in os.listdir(individual_dir):
        storylines.append(os.path.join(individual_dir, p))
        outdir = os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', 'populate_outputs', p.split('.')[0])
        outdirs.append(outdir)
    run_id = datetime.datetime.now().isoformat().replace(':', '-').split('.')[0]
    run_swg_mp(storyline_paths=storylines, outdirs=outdirs, ns=1,
               vcfs=vcfs, base_dirs=base_dirs,
               log_path=os.path.join(log_dir, 'generate_SWG_output_support_{}.txt'.format(run_id)),
               pool_size=1)


def generate_all_swg(n, n_is_final, outdir, vcfs=default_vcf, base_dirs=default_base_dir,
                     prob_path=os.path.join(climate_shocks_env.supporting_data_dir, 'prob_gen_event_swg.csv')):
    """
    genterate a full suite of values
    :param n: integer the number to run (see n_is_final)
    :param n_is_final: bool, if True then run enough runs that statistically n== number after the run if False then
                       simply run n runs
    :param outdir: directory to save the data
    :param base_dirs: base_dirs can be just single path
    :param vcfs: vcfs can be single path
    :param prob_path: path to read the probability data from, only used if n_is_final=True
    :return:
    """
    assert isinstance(n, int), 'n must be an int'
    assert isinstance(n_is_final, bool)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if n_is_final:
        prob_data = pd.read_csv(prob_path, index_col=0)
        if prob_data.isna().values.any():
            raise ValueError('null data in prob data, this should not be, check: {}'.format(prob_path))
        prob_data = prob_data.loc[:, '0'].to_dict()

    print('running SWG')
    storylines = []
    ns = []
    outdirs = []
    for p in os.listdir(individual_dir):
        if n_is_final:
            temp = int(n / (1 - prob_data[p.replace('.csv', '')]) + 1)
            ns.append(temp)
        else:
            ns.append(n)
        storylines.append(os.path.join(individual_dir, p))
        o = os.path.join(outdir, p.split('.')[0])
        outdirs.append(o)
    run_id = datetime.datetime.now().isoformat().replace(':', '-').split('.')[0]
    run_swg_mp(storyline_paths=storylines, outdirs=outdirs, ns=ns,
               vcfs=vcfs, base_dirs=base_dirs,
               log_path=os.path.join(log_dir, 'generate_all_swg_{}.txt'.format(run_id)))


def clean_individual(ind_dir, duplicate=True, merge=True):
    for p in os.listdir(ind_dir):
        if '.nc' in p:
            continue
        swg_dir = os.path.join(ind_dir, p)
        t = clean_swg(swg_dir=swg_dir, yml_path=os.path.join(swg_dir, 'ind.yml'),
                      duplicate=duplicate, merge=merge, nc_outpath=os.path.join(ind_dir, '{}_all.nc'.format(p)))
        print('{}: removed: {}'.format(p, len(t)))


if __name__ == '__main__':
    run = False  # to prevent accidental re-run
    if run:
        generate_SWG_output_support()
        full_dir = os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', 'full_SWG')
        generate_all_swg(10000, True, full_dir)
        clean_individual(full_dir)
