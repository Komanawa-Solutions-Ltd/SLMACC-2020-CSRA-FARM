"""
created matt_dumont 
on: 22/07/22
"""
import netCDF4 as nc
import numpy as np
import pandas as pd
from pathlib import Path
from Storylines.storyline_runs.run_random_suite import get_1yr_data

month_len = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def package_random(nc_files, irr_type, irr_mode):
    """

    :param nc_files: lsit of paths
    :param irr_type: list of irr type (bad, good)
    :param irr_mode: either dryland or irrigated (incl storage)
    :return:
    """
    mean_data = get_1yr_data(correct=False)
    mean_data.loc[:, 'kid'] = mean_data.loc[:, 'ID'] + '_' + mean_data.loc[:, 'irr_type'] + '_irr'
    mean_data.set_index('kid', inplace=True)
    pg_data = []
    sids = []
    probs = []
    expect_months = np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
    for i, (nc_file, irr) in enumerate(zip(nc_files, irr_type)):
        if i % 100 == 0:
            print(f'{i} of {len(nc_files)}')
        sid = Path(nc_file)
        sid = '-'.join(sid.name.split('-')[0:2]) + f'_{irr}_irr'
        data = nc.Dataset(nc_file)
        assert data.dimensions['sim_month'].size == 12
        assert (np.array(data.variables['m_month']) == expect_months).all()
        key = 'm_PGR'
        temp = np.array(data.variables[key]).transpose() * np.array([month_len[e] for e in expect_months])
        temp = np.concatenate((temp, temp.sum(axis=1)[:, np.newaxis]), axis=1)
        pg_data.append(temp.round())
        sids.extend(np.repeat(sid, len(temp)))
        probs.extend(np.repeat(mean_data.loc[sid, f'log10_prob_{irr_mode}'], len(temp)))
        data.close()
    pg_data = np.concatenate(pg_data)

    outdata = pd.DataFrame(columns=[f'pg_{m:02d}' for m in expect_months] + ['pg_1yr'], data=pg_data)
    outdata.loc[:, 'storyline'] = sids
    outdata.loc[:, f'log10_prob_{irr_mode}'] = probs

    # todo could manage dtypes if size is a problem.. move to ID, and boolean irr_type

    return outdata
    pass


# todo need to re-run probability across full suite otherwise it is wrong.
# todo need to update the get data to get mean data and get full dataset, both in this repo and in the final repo
# todo consider compression (which pandas handles natively)

if __name__ == '__main__':
    num = 1000
    test = package_random(['/home/matt_dumont/Downloads/rsl-000000-eyrewell-irrigated.nc' for e in range(num)],

                          ['bad' for e in range(num)],
                          'irrigated')
    print(test.dtypes)
    print('saving')
    test.to_hdf(Path().home().joinpath('Downloads/uncompressed.hdf'), 'test')
    test.to_hdf(Path().home().joinpath('Downloads/compressed_zlib5.hdf'), 'test', complevel=5)
    test.to_hdf(Path().home().joinpath('Downloads/compressed_zlib_9.hdf'), 'test', complevel=9)
    test.to_hdf(Path().home().joinpath('Downloads/compressed_lzo.hdf'), 'test', complevel=9, complib='lzo')
    test.to_hdf(Path().home().joinpath('Downloads/compressed_lz4.hdf'), 'test', complevel=9, complib='blosc:lz4')
