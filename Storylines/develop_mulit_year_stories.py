"""
 Author: Matt Hanson
 Created: 6/09/2021 10:16 AM
 """

import os
import pandas as pd
import numpy as np
from Storylines.storyline_runs.run_random_suite import get_1yr_data, default_mode_sites


def make_multi_year_stories_from_random_suite(outdir, year_stories, n, use_default_seed=True):
    """

    :param year_stories: dictionary with keys int (e.g. year 0,1,2....): vals list of paths to storylines
    :param n: number of storylines to create
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    assert isinstance(year_stories, dict)
    assert all([isinstance(k, int) for k in year_stories.keys()])
    if use_default_seed:
        np.random.seed(1156854)
        all_seeds = np.random.randint(1151, 16875324865, 5 * len(year_stories.keys()))
    else:
        all_seeds = np.random.randint(1151, 16875324865, 5 * len(year_stories.keys()))

    # set up all storyline data
    storyline_data = get_1yr_data(bad_irr=True, good_irr=True, correct=True)
    storyline_data.loc[:, 'full_ID'] = storyline_data.loc[:, 'ID'] + '_' + storyline_data.loc[:, 'irr_type']
    storyline_data.set_index('full_ID', inplace=True)
    # todo calc non exceedence probs!

    all_stories = []
    for v in year_stories.values():
        all_stories.extend(v)
    all_stories = np.array(all_stories)
    stories_exist = np.array([os.path.exists(s) for s in all_stories])
    assert all(stories_exist), 'stories do not exist:\n' + '\n'.join(all_stories[~stories_exist])
    storyline_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_stories]
    idx = np.in1d(storyline_ids, storyline_data.index)
    assert idx.all(), 'missing storyline ids:\n' + '\n'.join(storyline_ids[~idx])

    # initalise outdata
    columns = [f'rid_yr_{k:01d}' for k in year_stories.keys()] + ['true_prob_irr', 'true_prob_dry'] + \
              [f'non_exceed_prob_{site}-{mode}' for mode, site in default_mode_sites]
    outdata = pd.DataFrame(index=range(n), columns=columns)

    # set up random suite
    for i, (y, stories) in enumerate(year_stories.items()):
        np.random.seed(i)
    # todo random selection


    # todo save a params textfile...make sure it does not conflict with reading the storylines
    outdata.to_csv(os.path.join(outdir, 'id_probs'))  # todo may need to be a csv to avoid being read as a storyline
    return outdata


def create_1y_pg_data(bad_irr=True):  # todo work through for new system
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
