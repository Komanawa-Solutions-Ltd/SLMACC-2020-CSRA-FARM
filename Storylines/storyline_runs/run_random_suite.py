"""
 Author: Matt Hanson
 Created: 24/02/2021 10:46 AM
 """
import os
import ksl_env
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Storylines.generate_random_storylines import generate_random_suite
from BS_work.IID.IID import run_IID
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp, default_pasture_growth_dir, pgm_log_dir, \
    default_mode_sites

random_pg_dir = os.path.join(default_pasture_growth_dir, 'random')
random_sl_dir = os.path.join(temp_storyline_dir, 'random')
gdrive_outdir = os.path.join(ksl_env.slmmac_dir, 'random_stories_prob')

for d in [random_pg_dir, random_sl_dir, gdrive_outdir]:
    if not os.path.exists(d):
        os.makedirs(d)


def make_1_year_storylines():
    n = 70000  # based on an arbirary 4 day run length over easter
    storylines = generate_random_suite(n, use_default_seed=True, save=False, return_story=True)

    # run IID
    iid_prob = run_IID(story_dict={f'rsl-{k:06d}': v for k, v in enumerate(storylines)}, verbose=False)
    iid_prob.set_index('ID')
    iid_prob.to_hdf(os.path.join(random_pg_dir, 'IID_probs_1yr.hdf'), 'prob', mode='w')  # save locally
    iid_prob.to_hdf(os.path.join(gdrive_outdir, 'IID_probs_1yr.hdf'), 'prob', mode='w')  # save on gdrive

    # save non-zero probability stories
    for sl, (i, p) in zip(storylines, iid_prob.log10_prob.to_dict().items()):
        if not np.isfinite(p):
            continue
        name = f'rsl-{i:06d}'
        sl.to_csv(os.path.join(random_sl_dir, f'{name}.csv'))


def run_1year_basgra():
    run_stories = glob.glob(os.path.join(random_sl_dir, 'rsl-*.csv'))
    outdirs = [random_pg_dir for e in run_stories]
    run_full_model_mp(
        storyline_path_mult=run_stories,
        outdir_mult=outdirs,
        nsims_mult=100,
        log_path=os.path.join(pgm_log_dir, 'random'),
        description_mult='random 1 year storylines, see Storylines/generate_random_storylines.py and '
                         'Storylines/storyline_runs/run_random_suite.py for details',
        padock_rest_mult=False,
        save_daily_mult=False,
        verbose=False
    )


def create_1y_pg_data():
    # Autumn Drought Cumulative-rest-50-75-eyrewell-irrigated.nc
    data = pd.read_hdf(os.path.join(random_pg_dir, 'IID_probs_1yr.hdf'), 'prob')
    assert isinstance(data, pd.DataFrame)
    for site, mode in default_mode_sites:
        key = f'{mode}-{site}'
        data.loc[:, f'{key}_yr1'] = np.nan
        for i, idv in data.loc[:, ['ID']].itertuples(True, None):
            p = os.path.join(random_pg_dir, f'{idv}-{key}.nc')
            if not os.path.exists(p):
                continue

            nc_data = nc.Dataset(p)
            data.loc[i, f'{key}_yr1'] = np.array(nc_data.variables['m_PGRA_cum'][-1, :]).mean()
            nc_data.close()
    data.to_hdf(os.path.join(random_pg_dir, 'IID_probs_pg_1y.hdf'), 'prob',
                mode='w')
    data.to_hdf(os.path.join(gdrive_outdir, 'IID_probs_pg_1y.hdf'), 'prob',
                mode='w') #todo add cumulative production

#todo add better than median restrictions

def create_3yr_suite(use_default_seed=True, save_to_gdrive=True): #todo check will small suite
    n = 10  # todo how many, check this after running.
    data_1y = pd.read_hdf(os.path.join(random_pg_dir, 'IID_probs_pg_1y.hdf'), 'prob')
    assert isinstance(data_1y, pd.DataFrame)
    data_1y = data_1y.dropna()

    if use_default_seed:
        seed = 496148
    else:
        seed = np.random.randint(1, 500000)

    for site, mode in default_mode_sites:
        key = f'{site}-{mode}'
        np.random.seed(seed)
        idxs = np.random.randint(len(data_1y), size=(n * 3))
        outdata = pd.DataFrame(index=range(n), columns=['scen1', 'scen2', 'scen3',
                                                        'pga1', 'pga2', 'pga3',
                                                        'cpga2', 'cpga3', 'prob'])
        prob = data_1y['log10_prob'].values[idxs].reshape((n, 3))
        outdata.loc[:, 'log10_prob'] = prob.sum(
            axis=1)  # note that I have changed the probability to be log10(probaility)
        pga = data_1y[f'{key}_yr1'].values[idxs].reshape(n, 3)
        outdata.loc[:, 'pga1'] = pga[:, 0]
        outdata.loc[:, 'pga2'] = pga[:, 1]
        outdata.loc[:, 'pga3'] = pga[:, 2]
        outdata.loc[:, 'cpga2'] = pga[:, 0:2].sum(axis=1)
        outdata.loc[:, 'cpga3'] = pga.sum(axis=1)

        temp = idxs.reshape((n, 3))
        outdata.loc[:, 'scen1'] = temp[:, 0]
        outdata.loc[:, 'scen2'] = temp[:, 1]
        outdata.loc[:, 'scen3'] = temp[:, 2]

        outdata.to_hdf(os.path.join(random_pg_dir, 'IID_probs_pg_3y.hdf'), 'prob', mode='w')
        if save_to_gdrive:
            outdata.to_hdf(os.path.join(gdrive_outdir, 'IID_probs_pg_3y.hdf'), 'prob', mode='w')


def get_3yr_suite():
    return pd.read_hdf(os.path.join(gdrive_outdir, 'IID_probs_pg_3y.hdf'), 'prob')


"""
    timeit_test(r'C:/Users/dumon/python_projects/SLMACC-2020-CSRA/Storylines/storyline_runs/run_random_suite.py',
                ('make_1_year_storylines',  # 16 stories 0.366s
                 # 'run_1year_basgra',  # 16 stories (full logical on dickie), 100 reals of 1 yr sim: 89.17002 seconds
                 'create_1y_pg_data',  # 16 stories 0.197s
                 ), n=100)  # 90s per 16 stories.
                 
    Memory: 10mb/16 stories, 1mb/16 stories in cloud
    based on my math the 1yr suite should be c. 4-5gb in size in the cloud and c. 40-50gb on disk.
    
    
"""


if __name__ == '__main__':
    t = input('are you sure you want to run this, it takes 4 days y/n')
    if t != 'y':
        raise ValueError('stopped re-running')

    make_1_year_storylines()
    #run_1year_basgra()
    create_1y_pg_data()
    pass
