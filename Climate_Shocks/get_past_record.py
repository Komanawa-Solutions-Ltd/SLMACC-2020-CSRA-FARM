"""
 Author: Matt Hanson
 Created: 26/11/2020 1:05 PM
 """
from Climate_Shocks.vcsn_pull import vcsn_pull_single_site
import ksl_env
import os
import pandas as pd


def get_vcsn_record():
    data, use_cords = vcsn_pull_single_site(lat=-43.358,
                                            lon=172.301,
                                            year_min=1972,
                                            year_max=2019,
                                            use_vars=('evspsblpot', 'rsds', 'tasmax', 'tasmin', 'pr'))
    data.rename(columns={'evspsblpot': 'pet', 'pr': 'rain',
                         'rsds': 'radn', 'tasmax': 'tmax', 'tasmin': 'tmin'}, inplace=True)
    print(use_cords)
    return data


def get_restriction_record(recalc=False):
    data_path = ksl_env.shared_drives(r"SLMACC_2020\WIL data\restriction_record.csv")
    if not recalc and os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0)
        return data

    raw_data_path = ksl_env.shared_drives(r"SLMACC_2020\WIL data\OSHB_WaimakRiverData_withRestrictionInfo.xlsx")
    data = pd.read_excel(raw_data_path).loc[:, ['OHB flow m3/s', 'Take rate', 'Day', 'Month', 'Year']]
    data = data.rename(
        columns={'OHB flow m3/s': 'flow', 'Take rate': 'take', 'Day': 'day', 'Month': 'month', 'Year': 'year'})
    data.loc[:, 'f_rest'] = 1 - data.loc[:, 'take'] / data.loc[:, 'take'].max()
    data = data.loc[(data.year <= 2019) & (data.year >= 1972)]
    data = data.groupby(['day', 'month', 'year']).mean().reset_index()

    strs = ['{}-{:02d}-{:02d}'.format(y, m, d) for y, m, d in data[['year', 'month', 'day']].itertuples(False, None)]
    data.loc[:, 'date'] = pd.Series(pd.to_datetime(strs))
    data.loc[:, 'doy'] = data.date.dt.dayofyear
    data.loc[data.f_rest < 0.001, 'f_rest'] = 0

    data = data.sort_values('date').reset_index()

    data.to_csv(data_path)
    return data

