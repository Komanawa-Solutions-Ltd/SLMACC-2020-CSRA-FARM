"""
 Author: Matt Hanson
 Created: 24/02/2021 10:46 AM
 """
import os
import ksl_env
import glob
import numpy as np
import pandas as pd
import gc
import netCDF4 as nc
import psutil
import itertools
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Storylines.generate_random_storylines import generate_random_suite
from BS_work.IID.IID import run_IID
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp, default_pasture_growth_dir, pgm_log_dir, \
    default_mode_sites
from Pasture_Growth_Modelling.full_model_implementation import add_pasture_growth_anaomoly_to_nc
from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched
from Storylines.storyline_evaluation.transition_to_fraction import corr_pg

name = 'random'
random_pg_dir = os.path.join(default_pasture_growth_dir, name)
random_sl_dir = os.path.join(temp_storyline_dir, name)
gdrive_outdir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', name)

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
    print('generating random suite')
    storylines = generate_random_suite(n, use_default_seed=True, save=False, return_story=True, bad_irr=bad_irr)

    # run IID
    print('calculating irrigated prob')
    iid_prob = run_IID(story_dict={f'rsl-{k:06d}': v for k, v in enumerate(storylines)}, verbose=True,
                       irr_prob_from_zero=False, add_irr_prob=True)
    iid_prob.set_index('ID', inplace=True)
    iid_prob.rename(columns={'log10_prob': 'log10_prob_irrigated'}, inplace=True)

    print('calculating dryland prob')
    temp = run_IID(story_dict={f'rsl-{k:06d}': v for k, v in enumerate(storylines)}, verbose=True,
                   irr_prob_from_zero=False, add_irr_prob=False).set_index('ID')
    iid_prob.loc[:, 'log10_prob_dryland'] = temp.loc[:, 'log10_prob']
    iid_prob.reset_index(inplace=True)

    iid_prob.to_hdf(os.path.join(f'{random_pg_dir}{tnm}', 'IID_probs_1yr.hdf'), 'prob', mode='w')  # save locally
    iid_prob.to_hdf(os.path.join(gdrive_outdir, f'IID_probs_1yr{tnm}.hdf'), 'prob', mode='w')  # save on gdrive

    # save non-zero probability stories
    print('saving storylines')
    for sl, (i, p) in zip(storylines, iid_prob.log10_prob_irrigated.to_dict().items()):
        if not np.isfinite(p):
            continue
        name = f'rsl-{i:06d}'
        sl.to_csv(os.path.join(f'{random_sl_dir}{tnm}', f'{name}.csv'))


def run_1year_basgra(bad_irr=True, start=0, end=None):
    """

    :param bad_irr: bool if True then create irrigation from 50-99th percentile if False 1-50th percentile
    :param start: int, start index
    :param end: None or int, end index, stories[start:end] will be run, helpful for chunking data
    :return:
    """
    if bad_irr:
        tnm = '_bad_irr'
    else:
        tnm = '_good_irr'
    run_stories = glob.glob(os.path.join(f'{random_sl_dir}{tnm}', 'rsl-*.csv'))
    if end is None:
        end = len(run_stories)
    assert isinstance(end, int) and isinstance(start, int)
    run_stories = run_stories[start:end]
    outdirs = [f'{random_pg_dir}{tnm}' for e in run_stories]
    run_full_model_mp(
        storyline_path_mult=run_stories,
        outdir_mult=outdirs,
        nsims_mult=96,
        log_path=os.path.join(pgm_log_dir, 'random'),
        description_mult='random 1 year storylines, see Storylines/generate_random_storylines.py and '
                         'Storylines/storyline_runs/run_random_suite.py for details',
        padock_rest_mult=False,
        save_daily_mult=False,
        verbose=False,
        mode_sites_mult=mode_sites,
        re_run=False,  # and additional safety
        seed=102576037
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
        for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]:
            data.loc[:, f'{key}_pg_m{m:02d}'] = np.nan

        for i, idv in data.loc[:, ['ID']].itertuples(True, None):
            if i % 1000 == 0:
                print(f'starting to read sim {i} for site: {site} and mode: {mode}')
            p = os.path.join(f'{random_pg_dir}{tnm}', f'{idv}-{key}.nc')
            if not os.path.exists(p):
                continue

            nc_data = nc.Dataset(p)
            temp = np.array(nc_data.variables['m_PGR'])
            temp *= np.array([31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30])[:, np.newaxis]
            temp = np.nanmean(temp, axis=1)
            for j, m in enumerate([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]):
                data.loc[i, f'{key}_pg_m{m:02d}'] = temp[j]
            temp = temp.sum()
            data.loc[i, f'{key}_pg_yr1'] = temp  #
            nc_data.close()
    data.loc[:, 'irr_type'] = tp
    data.to_hdf(os.path.join(f'{random_pg_dir}{tnm}', f'IID_probs_pg_1y{tnm}.hdf'), 'prob',
                mode='w')
    data.to_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y{tnm}.hdf'), 'prob',
                mode='w')


def get_1yr_data(bad_irr=True, good_irr=True, correct=False):
    """
    get the 1 year data
    :param bad_irr: bool if True return the data from the worse than median irrigation restriction suite
    :param good_irr: bool if True return the data from the better than median irrigation restriction suite
    :param correct: bool if True add a correction, if False then use model output.
    :return:
    """
    assert any([bad_irr, good_irr])
    good, bad = None, None
    if bad_irr:
        bad = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_bad_irr.hdf'), 'prob')

    if good_irr:
        good = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_good_irr.hdf'), 'prob')

    if correct:
        data = pd.concat([good, bad])
        data = corr_pg(data, mode_site=mode_sites)
        return data
    else:
        return pd.concat([good, bad])


def create_nyr_suite(nyr, use_default_seed=True,
                     save_to_gdrive=True, correct=False, monthly_data=False):
    """
    this does keep the number consitant across sims as the same seed it used
    :param nyr: number of years long, options are: [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    :param use_default_seed: bool if true then use the default seed to keep reproducability
    :param save_to_gdrive: bool if True then save to the google drive
    :param correct: bool if True add a correction, if False then use model output.
    :param monthly_data: bool if true save monthly data, else save annual data.
    :return:
    """
    print(f'nyear: {nyr}')
    assert isinstance(nyr, int)
    n = int(2.5e8)
    if monthly_data:
        n = int(2.5e7)
    if nyr > 5:
        n = int(2.5e8)

    data_1y = get_1yr_data(bad_irr=True, good_irr=True, correct=correct)
    data_1y.reset_index(inplace=True)
    data_1y.loc[:, 'ID'] = data_1y.loc[:, 'ID'] + '-' + data_1y.loc[:, 'irr_type']
    data_1y.set_index('ID', inplace=True)
    assert isinstance(data_1y, pd.DataFrame)
    data_1y = data_1y.dropna()

    default_seeds = {
        1: 654654,
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

    mem = psutil.virtual_memory().available - 3e9  # leave 3 gb spare
    total_mem_needed = np.zeros(1).nbytes * n * nyr * 4
    if monthly_data:
        total_mem_needed *= 12
    chunks = int(np.ceil(total_mem_needed / mem))
    print(f'running in {chunks} chunks')
    chunk_size = int(np.ceil(n / chunks))

    for mode, site in default_mode_sites:
        print('making dataframe')
        outdata = pd.DataFrame(index=range(n), columns=['log10_prob_dryland', 'log10_prob_irrigated',
                                                        f'{site}-{mode}_pg_yr{nyr}'
                                                        ], dtype=np.float32)
        print(outdata.dtypes)
        print('/n', mode, site)
        key = f'{site}-{mode}'
        np.random.seed(seed)

        if 'store' in mode:
            temp_p = 10 ** data_1y.loc[:, f'log10_prob_irrigated']
        else:
            temp_p = 10 ** data_1y.loc[:, f'log10_prob_{mode}']
        p = temp_p / temp_p.sum()
        idxs = np.random.choice(
            np.arange(len(data_1y), dtype=np.uint32),
            size=(n * nyr),
            p=p
        ).reshape((n, nyr))

        for c in range(chunks):
            print(f'chunk: {c}')
            start_idx = chunk_size * c
            end_idx = chunk_size * (c + 1)
            cs = chunk_size
            if c == chunks - 1:
                end_idx = n
                cs = end_idx - start_idx
            print('getting prob_dry')
            prob = data_1y[f'log10_prob_dryland'].values[idxs[start_idx:end_idx]]
            # note that I have changed the probability to be log10(probaility)
            outdata.loc[start_idx:end_idx - 1, f'log10_prob_dryland'] = prob.sum(axis=1).astype(np.float32)

            print('getting prob_irr')
            prob = data_1y[f'log10_prob_irrigated'].values[idxs[start_idx:end_idx]]
            # note that I have changed the probability to be log10(probaility)
            outdata.loc[start_idx:end_idx - 1, f'log10_prob_irrigated'] = prob.sum(axis=1).astype(np.float32)

            print('getting pg')
            if monthly_data:
                for m in range(1, 13):
                    pga = data_1y[f'{key}_pg_m{m:02d}'].values[idxs[start_idx:end_idx]].reshape(cs, nyr)
                    for y in range(nyr):
                        outdata.loc[start_idx:end_idx - 1, f'{key}_pg_yr{y}_m{m:02d}'] = pga[:, y]
            else:
                pga = data_1y[f'{key}_pg_yr1'].values[idxs[start_idx:end_idx]].reshape(cs, nyr)
                outdata.loc[start_idx:end_idx - 1, f'{key}_pg_yr{nyr}'] = pga.sum(axis=1).astype(np.float32)

        if not use_default_seed:
            print('recording indexes')
            for n in range(nyr):
                outdata.loc[:, f'scen_{n + 1}'] = idxs[:, n]

        print(f'saving {mode} - {site} to local drive for {nyr}y')
        if monthly_data:
            if correct:
                dirnm = 'nyr_correct_monthly'
            else:
                dirnm = 'nyr_monthly'
        else:
            if correct:
                dirnm = 'nyr_correct'
            else:
                dirnm = 'nyr'

        outpath = os.path.join(os.path.dirname(random_pg_dir), dirnm, f'IID_probs_pg_{nyr}y_{site}-{mode}.npy')
        outpath_idx = os.path.join(os.path.dirname(random_pg_dir), dirnm, f'IID_stories_{nyr}y_{mode}.npy')

        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        np.save(outpath, outdata.values)
        with open(outpath.replace('.npy', '.csv'), 'w') as f:
            f.write(','.join(outdata.columns))
        if 'store' not in mode:
            np.save(outpath_idx, idxs)
        if save_to_gdrive:
            print(f'saving {mode} - {site} to google drive')
            if correct:
                outpath = os.path.join(gdrive_outdir, 'nyr_correct', f'IID_probs_pg_{nyr}y_{site}-{mode}.npy')
                outpath_idx = os.path.join(gdrive_outdir, 'nyr_correct',
                                           f'IID_stories_{nyr}y_{mode}.npy')
            else:
                outpath = os.path.join(gdrive_outdir, 'nyr', f'IID_probs_pg_{nyr}y_{site}-{mode}.npy')
                outpath_idx = os.path.join(gdrive_outdir, 'nyr', f'IID_stories_{nyr}y_{mode}.npy')

            np.save(outpath, outdata.values)
            if 'store' not in mode:
                np.save(outpath_idx, idxs)
            with open(outpath.replace('.npy', '.csv'), 'w') as f:
                f.write(','.join(outdata.columns))

        print(f'finished {mode} - {site}')
        gc.collect()
    if correct:
        outpath = os.path.join(os.path.dirname(random_pg_dir), 'nyr_correct', f'index_{nyr}y.csv')
    else:
        outpath = os.path.join(os.path.dirname(random_pg_dir), 'nyr', f'index_{nyr}y.csv')
    pd.Series(data_1y.index).to_csv(outpath)


def get_nyr_idxs(nyr, mode, correct=False, monthly_data=False):  # todo update with monthly
    extra = ''
    if monthly_data:
        extra = '_monthly'
    if correct:
        dir = os.path.join(os.path.dirname(random_pg_dir), f'nyr_correct{extra}')
    else:
        dir = os.path.join(os.path.dirname(random_pg_dir), f'nyr{extra}')

    if 'store' in mode:
        mode = 'irrigated'

    indexes = pd.read_csv(os.path.join(dir, f'index_{nyr}y.csv')).loc[:, 'ID'].values
    stories = np.load(os.path.join(dir, f'IID_stories_{nyr}y_{mode}.npy'))
    stories = indexes[stories]
    return stories


def get_nyr_suite(nyr, site, mode, correct=False, monthly_data=False):  # todo update with monthly
    """

    :param nyr:
    :param site:
    :param mode:
    :param correct: dnz correction applied
    :return:
    """
    extra = ''
    if monthly_data:
        extra = '_monthly'
    if correct:
        dir = os.path.join(os.path.dirname(random_pg_dir), f'nyr_correct{extra}')
    else:
        dir = os.path.join(os.path.dirname(random_pg_dir), f'nyr{extra}')

    outpath = os.path.join(dir,
                           f'IID_probs_pg_{nyr}y_{site}-{mode}.npy')
    out = np.load(outpath)
    out = pd.DataFrame(out, columns=pd.read_csv(outpath.replace('.npy', '.csv')).columns)
    return out


"""
    timeit_test(r'C:/Users/dumon/python_projects/SLMACC-2020-CSRA/Storylines/storyline_runs/run_random_suite.py',
                ('make_1_year_storylines',  # 16 stories 0.366s
                 # 'run_1year_basgra',  # 16 stories (full logical on dickie), 100 reals of 1 yr sim: 89.17002 seconds
                 'create_1y_pg_data',  # 16 stories 0.197s
                 ), n=100)  # 90s per 16 stories.
                 
    Memory: 10mb/16 stories, 1mb/16 stories in cloud
    based on my math the 1yr suite should be c. 4-5gb in size in the cloud and c. 40-50gb on disk.
    
    
"""


def fix_old_1yr_runs(base_dir, change_storyline_time=False):
    paths = glob.glob(os.path.join(base_dir, '*.nc'))
    pl = len(paths)
    for i, p in enumerate(paths):
        if i % 1000 == 0:
            print(f'{i} of {pl}')
        if change_storyline_time:
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


mode_sites = default_mode_sites

if __name__ == '__main__':
    run_chunks = []
    make_stories = False
    run_basgra_bad = True
    run_basgra_good = True
    extract_data_bad = True
    extract_data_good = True
    start = 0
    end = None
    if make_stories:
        make_1_year_storylines(bad_irr=True)
        make_1_year_storylines(bad_irr=False)

    if run_basgra_bad or run_basgra_good:
        t = input('are you sure you want to run this, it takes 8 days to run basgra y/n')
        if t != 'y':
            raise ValueError('stopped re-running')

    if run_basgra_bad:
        import time

        run_1year_basgra(bad_irr=True, start=start, end=end)

        t = time.time()

        print((time.time() - t) / 60, 'minutes to run 140k * 3  sims')
    if run_basgra_good:
        import time

        run_1year_basgra(bad_irr=False, start=start, end=end)

        t = time.time()

        print((time.time() - t) / 60, 'minutes to run 140k * 3  sims')

    if extract_data_bad:
        create_1y_pg_data(bad_irr=True)
    if extract_data_good:
        create_1y_pg_data(bad_irr=False)

    get_1yr_data(True, True, True)
