"""
 Author: Matt Hanson
 Created: 26/11/2020 1:05 PM
 """
from Climate_Shocks.vcsn_pull import vcsn_pull_single_site, change_vcsn_units
import project_base
import os
import pandas as pd
import numpy as np
from Climate_Shocks.climate_shocks_env import event_def_path, supporting_data_dir

vcsn_keys = ('year', 'month', 'day', 'doy', 'pet', 'radn', 'tmax', 'tmin', 'rain')

restriction_keys = ('day', 'doy', 'f_rest', 'flow', 'month', 'take', 'year')

sites = ('eyrewell', 'oxford')


def _get_eyrewell_detrended(version):
    """
    quick funcation to support reading detrended data to calculate percentiles!
    :return:
    """
    if version == 1:
        data = pd.read_csv(os.path.join(os.path.dirname(event_def_path), 'detrended_vcsn_for_matt.csv'),
                           skiprows=3)
    elif version == 2:
        data = pd.read_csv(os.path.join(supporting_data_dir, 'do_not_delete', 'detrended_vcsn_for_matt_v2.csv'),
                           skiprows=3)
    else:
        raise NotImplementedError()
    data.loc[:, 'date'] = pd.to_datetime(data.loc[:, 'date'])
    data.loc[:, 'month'] = data.loc[:, 'date'].dt.month
    data.loc[:, 'year'] = data.loc[:, 'date'].dt.year
    data.loc[:, 'doy'] = data.loc[:, 'date'].dt.dayofyear
    data.set_index('date', inplace=True)
    data.sort_index(0, inplace=True)
    change_vcsn_units(data)
    data.rename(columns={'evspsblpot': 'pet', 'pr': 'rain',
                         'rsds': 'radn', 'tasmax': 'tmax', 'tasmin': 'tmin'}, inplace=True)

    return data


def get_vcsn_record(version='trended', site='eyrewell', recalc=False):
    if version == 'trended':
        pass
    elif version == 'detrended' and site == 'eyrewell':
        raise ValueError('depreciated')
        return _get_eyrewell_detrended(1)
    elif version == 'detrended2' and site == 'eyrewell':
        return _get_eyrewell_detrended(2)
    else:
        raise ValueError('incorrect {} for version'.format(version))

    if site == 'eyrewell':
        lat, lon = -43.372, 172.333
    elif site == 'oxford':
        lat, lon = -43.296, 172.192
    else:
        raise NotImplementedError('site: {} not implemented'.format(site))
    key = 'weather_data'
    data_path = ksl_env.slmmac_dir.joinpath(r"weather_date\{}.hdf".format(site))
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))

    if os.path.exists(data_path) and not recalc:
        data = pd.read_hdf(data_path, key=key)
        return data

    data, use_cords = vcsn_pull_single_site(lat,
                                            lon,
                                            year_min=1972,
                                            year_max=2019,
                                            use_vars=('evspsblpot', 'rsds', 'tasmax', 'tasmin', 'pr'))
    data.rename(columns={'evspsblpot': 'pet', 'pr': 'rain',
                         'rsds': 'radn', 'tasmax': 'tmax', 'tasmin': 'tmin'}, inplace=True)
    print(use_cords)
    data = data.set_index('date')
    data.to_hdf(data_path, key=key, mode='w')
    return data


def get_restriction_record(version='trended', recalc=False):
    if version == 'trended':
        data_path = os.path.join(os.path.dirname(event_def_path), 'restriction_record.csv')
        dt_format = '%Y-%m-%d'
        raw_data_path = ksl_env.slmmac_dir.joinpath("WIL data/OSHB_WaimakRiverData_withRestrictionInfo.xlsx")
    elif version == 'detrended' or version == 'detrended2':
        dt_format = '%Y-%m-%d'
        data_path = os.path.join(os.path.dirname(event_def_path), 'restriction_record_detrend.csv')
    else:
        raise ValueError('unexpected argument for version {} expected either trended or detrended'.format(version))

    if not recalc and os.path.exists(data_path):
        int_keys = {
            'day': int,
            'doy': int,
            'month': int,
            'year': int,
            'f_rest': float,
            'flow': float,
            'take': float,
        }
        data = pd.read_csv(data_path, dtype=int_keys)
        data.loc[:, 'date'] = pd.to_datetime(data.loc[:, 'date'], format=dt_format)
        data.set_index('date', inplace=True)
        data.sort_index(inplace=True)

        return data

    if version == 'trended':
        data = pd.read_excel(raw_data_path).loc[:, ['OHB flow m3/s', 'Take rate', 'Day', 'Month', 'Year']]
        data = data.rename(
            columns={'OHB flow m3/s': 'flow', 'Take rate': 'take', 'Day': 'day', 'Month': 'month', 'Year': 'year'})
        data.loc[:, 'f_rest'] = 1 - data.loc[:, 'take'] / data.loc[:, 'take'].max()
        data = data.loc[(data.year <= 2019) & (data.year >= 1972)]
        data = data.groupby(['day', 'month', 'year']).mean().reset_index()

        strs = ['{}-{:02d}-{:02d}'.format(y, m, d) for y, m, d in
                data[['year', 'month', 'day']].itertuples(False, None)]
        data.loc[:, 'date'] = pd.Series(pd.to_datetime(strs))
        data.loc[:, 'doy'] = data.date.dt.dayofyear
        data = data.set_index('date')

    elif version == 'detrended':
        raise ValueError('detrended record is calculated via another process')

    else:
        raise ValueError('unexpected argument for version {} expected either trended or detrended'.format(version))

    data.loc[data.f_rest < 0.001, 'f_rest'] = 0

    outdata = pd.DataFrame(index=pd.date_range('1972-01-01', '2019-12-31', name='date'), columns=['flow',
                                                                                                  'take',
                                                                                                  'day',
                                                                                                  'month',
                                                                                                  'year',
                                                                                                  'f_rest'])
    outdata.loc[:] = np.nan
    outdata = outdata.combine_first(data)

    outdata = outdata.fillna(method='ffill')
    outdata.loc[:, 'day'] = outdata.index.day
    outdata.loc[:, 'year'] = outdata.index.year
    outdata.loc[:, 'month'] = outdata.index.month
    outdata.loc[:, 'doy'] = outdata.index.dayofyear

    outdata.to_csv(data_path)
    return outdata


if __name__ == '__main__':
    trend = get_restriction_record()
    detrend = get_restriction_record('detrended')
    pass
