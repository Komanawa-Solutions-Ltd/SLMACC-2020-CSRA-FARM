"""
 Author: Matt Hanson
 Created: 13/07/2021 11:43 AM
 """
import shutil

import ksl_env
import os
import pandas as pd
from Storylines.storyline_building_support import month_len, default_mode_sites
from copy import deepcopy
import numpy as np
from Climate_Shocks.climate_shocks_env import temp_storyline_dir

dnz_values = {
    # oxford site : https://www.dairynz.co.nz/media/5793235/average-pasture-growth-data-south-island-2020-v1.pdf

    7: 8,
    8: 17,
    9: 39,
    10: 65,
    11: 68,
    12: 61,
    1: 72,
    2: 58,
    3: 51,
    4: 43,
    5: 28,
    6: 21,
}

historical_average = {

    7: 0.88089995,
    8: 4.712506795,
    9: 22.08360076,
    10: 43.26791787,
    11: 69.39725356,
    12: 81.39082945,
    1: 84.2771157,
    2: 66.55072322,
    3: 51.77521592,
    4: 38.16173217,
    5: 11.56814694,
    6: 4.467724458,

}

fixed_data = {
    ('dryland', 'oxford'): {
        6: 5 * 30,
        7: 5 * 31,
        8: 10 * 31
    },
    ('irrigated', 'eyrewell'): {
        6: 5 * 30,
        7: 10 * 31,
        8: 15 * 31
    },
    ('irrigated', 'oxford'): {
        6: 5 * 30,
        7: 5 * 31,
        8: 10 * 31
    },
}

deltas = {
    9: 1.4,
    10: 1.4,
    11: 1.,
    12: 0.8,
    1: 0.8,
    2: 0.8,
    3: 0.8,
    4: 1.,
    5: 1.,
}


def corr_pg(data, mode_site=default_mode_sites):
    """
    # note that fractions are of cumulative montly  data and that sum of probilities does not match the annual data
    :param data:
    :return:
    """
    data = deepcopy(data)

    for mode, site in mode_site:
        use_mode = mode
        if 'store' in use_mode:
            use_mode = 'irrigated'
        for m in range(1, 13):
            if m in [6, 7, 8]:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] = fixed_data[(use_mode, site)][m]
            else:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] *= deltas[m]

        # 1 year
        data.loc[:, f'{site}-{mode}_pg_yr1'] = data.loc[:,
                                               [f'{site}-{mode}_pg_m{m:02d}' for m in range(1, 13)]].sum(axis=1)

    return data


def corr_pg_raw(data, site, mode):
    use_mode = mode
    if 'store' in use_mode:
        use_mode = 'irrigated'
    for m in range(1, 13):
        if m in [6, 7, 8]:
            data.loc[:, f'pg_{m:02d}'] = fixed_data[(use_mode, site)][m]
        else:
            data.loc[:, f'pg_{m:02d}'] *= deltas[m]

    # 1 year
    data.loc[:, f'pg_yr1'] = data.loc[:, [f'pg_{m:02d}' for m in range(1, 13)]].sum(axis=1)

    return data


target_ranges = {
    ('dryland', 'oxford'): (4, 6 * 100000),
    ('irrigated', 'eyrewell'): (15 * 1000, 17 * 1000),
    ('irrigated', 'oxford'): (11.5 * 1000, 14 * 1000),  # todo need to update these for each new site, mode
}


def get_most_probabile(site, mode, correct=False):
    out = {}  # keys integer months and '1yr'
    gdrive_outdir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', 'random')
    bad = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_bad_irr.hdf'), 'prob')

    good = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_good_irr.hdf'), 'prob')

    data = pd.concat([good, bad])
    data = data.dropna()
    if 'store' in mode:
        use_mode = 'irrigated'  # most probible does not change much when you add the irrigation range...
    else:
        use_mode = mode
    minv, maxv = target_ranges[(use_mode, site)]
    data = data.loc[(minv <= data.loc[:, f'{site}-{mode}_pg_yr1']) & (data.loc[:, f'{site}-{mode}_pg_yr1'] <= maxv)]
    if correct:
        data = corr_pg(data)
    for m in range(1, 13):
        out[m] = data.loc[:, f'{site}-{mode}_pg_m{m:02d}'].mean()

    return out


def save_most_probable_storylines(outdir):
    os.makedirs(outdir, exist_ok=True)
    gdrive_outdir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', 'random')
    bad = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_bad_irr.hdf'), 'prob')

    good = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_good_irr.hdf'), 'prob')

    data = pd.concat([good, bad])
    data = data.dropna()
    for mode, site in target_ranges.keys():
        minv, maxv = target_ranges[(mode, site)]
        data = data.loc[(minv <= data.loc[:, f'{site}-{mode}_pg_yr1']) & (data.loc[:, f'{site}-{mode}_pg_yr1'] <= maxv)]

    name = 'random'
    random_sl_dir = os.path.join(temp_storyline_dir, name)
    for i, it in data.loc[:, ['ID', 'irr_type']].itertuples(False, None):
        src = os.path.join(f'{random_sl_dir}_{it}_irr', f'{i}.csv')
        dest = os.path.join(outdir, f'{i}_{it}_irr.csv')
        if not os.path.exists(dest):
            shutil.copyfile(src, dest)


if __name__ == '__main__':
    save_most_probable_storylines(r"C:\Users\dumon\Downloads\most_probable")

    raise NotImplementedError
    plot_months = [
        7,
        8,
        9,
        10,
        11,
        12,
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    outdir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', 'most_probable')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outdata = pd.DataFrame(index=plot_months)
    outdata.index.name = 'month'
    for mode, site in default_mode_sites:
        temp = get_most_probabile(site, mode, True)
        for m in plot_months:
            outdata.loc[m, f'{site}-{mode}'] = temp[m] / month_len[m]

    outdata.to_csv(os.path.join(outdir, 'most_probable_data.csv'))
