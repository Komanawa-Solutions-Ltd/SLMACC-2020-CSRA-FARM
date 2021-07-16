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


def corr_pg(data):
    """
    # note that fractions are of cumulative montly  data and that sum of probilities does not match the annual data
    :param data:
    :return:
    """
    data = deepcopy(data)

    for mode, site in default_mode_sites:
        for m in range(1, 13):
            if m in [6, 7, 8]:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] = fixed_data[(mode, site)][m]
            else:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] *= deltas[m]

        # 1 year
        data.loc[:, f'{site}-{mode}_pg_yr1'] = data.loc[:,
                                               [f'{site}-{mode}_pg_m{m:02d}' for m in range(1, 13)]].sum(axis=1)

    return data


if __name__ == '__main__':
    pass
