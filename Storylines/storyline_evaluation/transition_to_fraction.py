"""
 Author: Matt Hanson
 Created: 13/07/2021 11:43 AM
 """
import ksl_env
import os
import pandas as pd
from Storylines.storyline_building_support import month_len, default_mode_sites
from copy import deepcopy
import numpy as np

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


def get_most_probabile(site, mode):
    target_ranges = {
        ('dryland', 'oxford'): (4, 6 * 100000),
        ('irrigated', 'eyrewell'): (15 * 1000, 17 * 1000),
        ('irrigated', 'oxford'): (11.5 * 1000, 14 * 1000),
    }

    out = {}  # keys integer months and '1yr'
    gdrive_outdir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', 'random')
    bad = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_bad_irr.hdf'), 'prob')

    good = pd.read_hdf(os.path.join(gdrive_outdir, f'IID_probs_pg_1y_good_irr.hdf'), 'prob')

    data = pd.concat([good, bad])
    data = data.dropna()
    minv, maxv = target_ranges[(mode, site)]
    data = data.loc[(minv <= data.loc[:, f'{site}-{mode}_pg_yr1']) & (data.loc[:, f'{site}-{mode}_pg_yr1'] <= maxv)]
    data.loc[:, f'{site}-{mode}_pg_yr1'] += (2 * 300
                                             - data.loc[:, f'{site}-{mode}_pg_m06']
                                             - data.loc[:, f'{site}-{mode}_pg_m07'])
    out['1yr'] = data.loc[:, f'{site}-{mode}_pg_yr1'].mean()
    for m in range(1, 13):
        out[m] = data.loc[:, f'{site}-{mode}_pg_m{m:02d}'].mean()

    out[6] = 300
    out[7] = 300

    return out


def to_fract(data):
    """
    # note that fractions are of cumulative montly  data and that sum of probilities does not match the annual data
    :param data:
    :return:
    """
    data = deepcopy(data)

    for mode, site in default_mode_sites:
        data.loc[:, f'{site}-{mode}_pg_yr1'] += (2 * 300
                                                 - data.loc[:, f'{site}-{mode}_pg_m06']
                                                 - data.loc[:, f'{site}-{mode}_pg_m07'])

        data.loc[:, f'{site}-{mode}_pg_m06'] = 300
        data.loc[:, f'{site}-{mode}_pg_m07'] = 300

        divisor = get_most_probabile(site, mode)

        for m in range(1, 13):
            data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] *= 1 / divisor[m]

        # 1 year
        data.loc[:, f'{site}-{mode}_pg_yr1'] *= 1 / divisor['1yr']

    return data
