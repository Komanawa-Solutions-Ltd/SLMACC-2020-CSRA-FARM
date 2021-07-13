"""
 Author: Matt Hanson
 Created: 13/07/2021 11:43 AM
 """
from Storylines.storyline_building_support import month_len, default_mode_sites
from copy import deepcopy

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

default_vals = {
    'oxford': {
        6: 10.5,
        7: 4.,
    },
    'eyrewell': {
        6: 21.,
        7: 8.,
    },
}


def correct_for_DNZ(data):
    data = deepcopy(data)
    for mode, site in default_mode_sites:
        for m in range(1, 13):
            if m in [6, 7]:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] = default_vals[site][m]
            else:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] *= dnz_values[m] / historical_average[m]

        # 1 year
        data.loc[:, f'{site}-{mode}_pg_yr1'] = data.loc[:, [f'{site}-{mode}_pg_m{m:02d}' for m in range(1, 13)]].sum(axis=1)
    return data
