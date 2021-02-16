"""
 Author: Matt Hanson
 Created: 27/01/2021 9:52 AM
 """
import pandas as pd
from Storylines.check_storyline import ensure_no_impossible_events
from Climate_Shocks import climate_shocks_env
import os

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


def get_baseline_storyline(save=False):
    rest_data = get_rest_base_data()
    for i in [5, 6, 7, 8]:
        rest_data[i] = 0
    data = pd.DataFrame(index=pd.date_range('2025-07-01', '2028-06-01', freq='MS'),
                        columns=['precip_class', 'temp_class', 'rest'])
    data.index.name = 'date'
    data.loc[:, 'year'] = data.index.year
    data.loc[:, 'month'] = data.index.month
    for i, y, m in data.loc[:, ['year', 'month']].itertuples(True, None):
        data.loc[i, 'precip_class'] = 'A'
        data.loc[i, 'temp_class'] = 'A'
        data.loc[i, 'rest'] = 0  # todo what to do with restrictions detrend the restriction data and use mean?
        if m == 7:
            data.loc[i, 'temp_class'] = 'C'  # no average temp julys

    data.loc[:, 'rest'] = data.loc[:, 'month'].replace(rest_data)
    ensure_no_impossible_events(data)
    if save:
        data.to_csv(os.path.join(climate_shocks_env.storyline_dir, '0-baseline.csv'))

    return data


def get_rest_base_data():
    data = pd.read_csv(climate_shocks_env.event_def_path, skiprows=1)
    data.loc[:, 'mlen'] = data.loc[:, 'month'].replace(month_len)
    data.loc[:, 'rest'] = data.loc[:, 'rest_cum'] / data.loc[:, 'mlen']
    data = data.loc[(data.temp == 0) & (data.precip == 0)]
    out = data.loc[:, ['month', 'rest']].groupby('month').describe().round(4)

    return out[('rest', 'mean')].to_dict()


if __name__ == '__main__':
    get_baseline_storyline(True)
