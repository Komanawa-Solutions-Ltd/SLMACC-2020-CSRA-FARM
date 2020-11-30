"""
 Author: Matt Hanson
 Created: 20/10/2020 9:06 AM
 """

import numpy as np
import pandas as pd
import os
import netCDF4 as nc
import ksl_env
import glob


def change_vcsn_units(data):
    for k in data.keys():
        if k in ['tasmax', 'tasmin']:
            # convert to C.
            # originally in k
            data.loc[:, k] += - 273.15
        elif k in ['evspsblpot', 'pr', 'pradj']:
            # convert to mm
            # orginally in kg/m2/s or mm/m2/s
            # kg/m2 ==mm/m2
            data.loc[:, k] *= 86400
        elif k == 'rsds':
            # convert to MJ/m2/d
            # orignally in W/m2
            data.loc[:, k] *= 86400 * 1e-6
        else:
            continue


def vcsn_pull_single_site(lat, lon, year_min, year_max, use_vars='all', vcsn_dir=ksl_env.get_vscn_dir()):
    """
    pull vcsn data where all vars in each file, but each file is 1 year  return pandas dataframe
    :param vcsn_dir: where .nc files are
    :param lat: site lat
    :param lon: site lon
    :param year_min: first year to include
    :param year_max: last year to include
    :param use_varsvars: 'all' or variables to pull, should be one of:
                 ('evspsblpot', 'pr', 'pradj', 'rsds', 'tasmax', 'tasmin')
    :return: data(pd.DataFrame), (use_lat, use_lon)
    """
    # pradj excluded as it is weird, look into this and ask daithi.  Before 1997 pradj looks like a temp variable
    all_vars = ('evspsblpot', 'pr', 'rsds', 'tasmax', 'tasmin')
    if use_vars == 'all':
        use_vars = all_vars
    else:
        assert np.in1d(use_vars, all_vars).all(), 'unknown variables, expected only: {}'.format(all_vars)

    assert os.path.exists(vcsn_dir), 'vscns dir does not exist'

    # initialize data
    all_data = {}
    all_data['date'] = []
    for v in use_vars:
        all_data[v] = []

    for yr in range(year_min, year_max + 1):
        path = os.path.join(vcsn_dir, 'vcsn_{y}0101-{y}1231_for-Komanawa.nc'.format(y=yr))
        data = nc.Dataset(path)

        lon_idx = np.argmin(np.abs(np.array(data.variables['lon'][:]) - lon))
        use_lon = np.array(data.variables['lon'])[lon_idx]

        lat_idx = np.argmin(np.abs(np.array(data.variables['lat'][:]) - lat))
        use_lat = np.array(data.variables['lat'])[lat_idx]

        all_data['date'].extend(np.array(data.variables['date']))

        for v in use_vars:
            all_data[v].extend(np.array(data.variables[v][:, lat_idx, lon_idx]))

    out_data = pd.DataFrame(all_data)
    out_data.loc[:, 'date'] = pd.to_datetime(out_data.loc[:, 'date'], format='%Y%m%d')
    out_data.loc[:, 'year'] = out_data.date.dt.year
    out_data.loc[:, 'month'] = out_data.date.dt.month
    out_data.loc[:, 'day'] = out_data.date.dt.day
    out_data.loc[:, 'doy'] = out_data.date.dt.dayofyear
    out_data = out_data.loc[:, ['date', 'year', 'month', 'day', 'doy'] + list(use_vars)]

    change_vcsn_units(out_data)
    return out_data, (use_lat, use_lon)


if __name__ == '__main__':
    print('west eyreton')
    lat, lon = -43.34104969510804, 172.32893676842548
    out, (use_lat, use_lon) = vcsn_pull_single_site(
                                                    lat=lat,
                                                    lon=lon,
                                                    year_min=1972,
                                                    year_max=2019, )
    out = out.groupby('year').sum()
    print(use_lat, use_lon)
    print(out.pr.mean())
    print(out.pr.std())

    print('oxford')
    lat, lon = -43.29259008790322, 172.19624253342405
    out2, (use_lat, use_lon) = vcsn_pull_single_site(
                                                    lat=lat,
                                                    lon=lon,
                                                    year_min=1972,
                                                    year_max=2019, )
    out2 = out2.groupby('year').sum()
    print(out2.pr.mean())
    print(out2.pr.std())
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, sharex=True)
    ax.hist(out['pr'], color='r', bins=20)
    ax.set_title('eyrewell')
    ax2.hist(out2['pr'], color='b', bins=20)
    ax2.set_title('oxford')
    print('ttest')
    from scipy.stats import ttest_ind
    print(ttest_ind(out.pr,out2.pr))
    plt.show()
    pass