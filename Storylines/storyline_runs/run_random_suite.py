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
import itertools
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Storylines.generate_random_storylines import generate_random_suite
from BS_work.IID.IID import run_IID
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp, default_pasture_growth_dir, pgm_log_dir, \
    default_mode_sites
from Pasture_Growth_Modelling.full_model_implementation import add_pasture_growth_anaomoly_to_nc

random_pg_dir = os.path.join(default_pasture_growth_dir, 'random')
random_sl_dir = os.path.join(temp_storyline_dir, 'random')
gdrive_outdir = os.path.join(ksl_env.slmmac_dir, 'random_stories_prob')

for d, tnm in itertools.product([random_pg_dir, random_sl_dir], ['_bad_irr', '_good_irr']):
    if not os.path.exists(f'{d}{tnm}'):
        os.makedirs(f'{d}{tnm}')

if not os.path.exists(gdrive_outdir):
    os.makedirs(gdrive_outdir)


def make_1_year_storylines(bad_irr=True):
    """

    :param bad_irr: bool if True then create irrigation from 50-99th percentile if False 1-50th percentile
    :return:
    """
    if bad_irr:
        tnm = '_bad_irr'
    else:
        tnm = '_good_irr'
    n = 70000  # based on an arbirary 4 day run length over easter
    storylines = generate_random_suite(n, use_default_seed=True, save=False, return_story=True, bad_irr=bad_irr)

    # run IID
    iid_prob = run_IID(story_dict={f'rsl-{k:06d}': v for k, v in enumerate(storylines)}, verbose=False)
    iid_prob.set_index('ID')
    iid_prob.to_hdf(os.path.join(f'{random_pg_dir}{tnm}', 'IID_probs_1yr.hdf'), 'prob', mode='w')  # save locally
    iid_prob.to_hdf(os.path.join(gdrive_outdir, f'IID_probs_1yr{tnm}.hdf'), 'prob', mode='w')  # save on gdrive

    # save non-zero probability stories
    for sl, (i, p) in zip(storylines, iid_prob.log10_prob.to_dict().items()):
        if not np.isfinite(p):
            continue
        name = f'rsl-{i:06d}'
        sl.to_csv(os.path.join(f'{random_sl_dir}{tnm}', f'{name}.csv'))


def run_1year_basgra(bad_irr=True):
    """

    :param bad_irr: bool if True then create irrigation from 50-99th percentile if False 1-50th percentile
    :return:
    """
    if bad_irr:
        tnm = '_bad_irr'
    else:
        tnm = '_good_irr'
    run_stories = glob.glob(os.path.join(f'{random_sl_dir}{tnm}', 'rsl-*.csv'))
    outdirs = [f'{random_pg_dir}{tnm}' for e in run_stories]
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


def create_1y_pg_data(bad_irr=True):
    """

    :param bad_irr: bool if True then create irrigation from 50-99th percentile if False 1-50th percentile
    :return:
    """
    if bad_irr:
        tnm = '_bad_irr'
        tp = 'bad'
    else:
        tp = 'good'
        tnm = '_good_irr'  # Autumn Drought Cumulative-rest-50-75-eyrewell-irrigated.nc
    data = pd.read_hdf(os.path.join(f'{random_pg_dir}{tnm}', 'IID_probs_1yr.hdf'), 'prob')
    assert isinstance(data, pd.DataFrame)
    for site, mode in default_mode_sites:
        key = f'{mode}-{site}'
        data.loc[:, f'{key}_pg_yr1'] = np.nan
        data.loc[:, f'{key}_pgra_yr1'] = np.nan
        for i, idv in data.loc[:, ['ID']].itertuples(True, None):
            if i % 1000 == 0:
                print(f'starting to read sim {i} for site: {site} and mode: {mode}')
            p = os.path.join(f'{random_pg_dir}{tnm}', f'{idv}-{key}.nc')
            if not os.path.exists(p):
                continue

            nc_data = nc.Dataset(p)
            data.loc[i, f'{key}_pgra_yr1'] = np.array(nc_data.variables['m_PGRA_cum'][-1, :]).mean()
            temp = np.array(nc_data.variables['m_PGR'])
            temp *= np.array([31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30])[:, np.newaxis]
            temp = temp.sum(axis=0).mean()
            data.loc[i, f'{key}_pg_yr1'] = temp
            nc_data.close()
    data.loc[:, 'irr_type'] = tp
    data.to_hdf(os.path.join(f'{random_pg_dir}{tnm}', f'IID_probs_pg_1y{tnm}.hdf'), 'prob',
                mode='w')
    data.to_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y{tnm}.hdf'), 'prob',
                mode='w')


def get_1yr_data(bad_irr=True, good_irr=True):
    """
    get the 1 year data
    :param bad_irr: bool if True return the data from the worse than median irrigation restriction suite
    :param good_irr: bool if True return the data from the better than median irrigation restriction suite
    :return:
    """
    assert any([bad_irr, good_irr])
    good, bad = None, None
    if bad_irr:
        bad = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_bad_irr.hdf'), 'prob')

    if good_irr:
        good = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_good_irr.hdf'), 'prob')

    return pd.concat([good, bad])


def create_nyr_suite(nyr, use_default_seed=True,
                     save_to_gdrive=True):
    """

    :param nyr: number of years long, options are: [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    :param use_default_seed: bool if true then use the default seed to keep reproducability
    :param save_to_gdrive: bool if True then save to the google drive
    :return:
    """
    assert isinstance(nyr, int)
    # todo check will small suite
    # todo i need to add rest per to all storylines !!!
    n = int(5e8)  # todo finalize should yield a 16gb file
    data_1y = get_1yr_data(bad_irr=True, good_irr=True)
    assert isinstance(data_1y, pd.DataFrame)
    data_1y = data_1y.dropna()
    default_seeds = {
        2: 471121,
        3: 44383,
        4: 80942,
        5: 464015,
        6: 246731,
        7: 229599,
        8: 182848,
        9: 310694,
        10: 367013,
        15: 458445
    }
    if use_default_seed:
        seed = default_seeds[nyr]
    else:
        seed = np.random.randint(1, 500000)

    outdata = pd.DataFrame(index=range(n))
    for mode, site in default_mode_sites:
        key = f'{site}-{mode}'
        np.random.seed(seed)
        idxs = np.random.randint(len(data_1y), size=(n * nyr))
        prob = data_1y['log10_prob'].values[idxs].reshape((n, nyr))
        outdata.loc[:, 'log10_prob'] = prob.sum(
            axis=1)  # note that I have changed the probability to be log10(probaility)
        pga = data_1y[f'{key}_pgra_yr1'].values[idxs].reshape(n, nyr)
        outdata.loc[:, f'{key}_cpg'] = pga.sum(axis=1)

        pga = data_1y[f'{key}_pg_yr1'].values[idxs].reshape(n, nyr)
        outdata.loc[:, f'{key}_cpg'] = pga.sum(axis=1)

    if not use_default_seed:
        temp = idxs.reshape((n, nyr))
        for n in range(nyr):
            outdata.loc[:, f'scen{n + 1}'] = temp[:, n]

    outdata.to_hdf(os.path.join(os.path.dirname(random_pg_dir), f'IID_probs_pg_{nyr}y.hdf'), 'prob',
                   mode='w')
    if save_to_gdrive:
        outdata.to_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_{nyr}y.hdf'), 'prob', mode='w')


def get_nyr_suite(nyr):
    return pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_{nyr}y.hdf'), 'prob')


"""
    timeit_test(r'C:/Users/dumon/python_projects/SLMACC-2020-CSRA/Storylines/storyline_runs/run_random_suite.py',
                ('make_1_year_storylines',  # 16 stories 0.366s
                 # 'run_1year_basgra',  # 16 stories (full logical on dickie), 100 reals of 1 yr sim: 89.17002 seconds
                 'create_1y_pg_data',  # 16 stories 0.197s
                 ), n=100)  # 90s per 16 stories.
                 
    Memory: 10mb/16 stories, 1mb/16 stories in cloud
    based on my math the 1yr suite should be c. 4-5gb in size in the cloud and c. 40-50gb on disk.
    
    
"""


def fix_old_1yr_runs(base_dir):
    paths = glob.glob(os.path.join(base_dir, '*.nc'))
    pl = len(paths)
    for i, p in enumerate(paths):
        if i % 1000 == 0:
            print(f'{i} of {pl}')
        data = nc.Dataset(p, mode='a')
        # change years
        data.variables['m_year'][:] = np.array([2025, 2025, 2025, 2025, 2025, 2025, 2026, 2026, 2026,
                                                2026, 2026, 2026]) - 1
        # add some metadata that a change happened in the description
        data.description = data.description + (' storyline changed with fix_old_1yr_runs to '
                                               'shift storyline start to july 2024 from 2025')
        # fix storyline
        data.storyline = [e.replace('2025', '2024').replace('2026', '2025') for e in data.storyline]

        data.close()

        # re-run add pgra
        add_pasture_growth_anaomoly_to_nc(p)


if __name__ == '__main__':
    t = input('are you sure you want to run this, it takes 4 days y/n')
    if t != 'y':
        raise ValueError('stopped re-running')
    # only run next line of code once as this fixes a mistake from previously
    # fix_old_1yr_runs(r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\random_bad_irr")

    make_1_year_storylines(bad_irr=True)
    # run_1year_basgra(bad_irr=True)
    create_1y_pg_data(bad_irr=True)

    make_1_year_storylines(bad_irr=False)
    # run_1year_basgra(bad_irr=False)
    create_1y_pg_data(bad_irr=False)

    pass
