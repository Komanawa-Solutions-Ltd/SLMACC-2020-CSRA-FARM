"""
 Author: Matt Hanson
 Created: 27/01/2021 9:52 AM
 """
import pandas as pd
from Storylines.check_storyline import ensure_no_impossible_events
from Climate_Shocks import climate_shocks_env
import os
from Storylines.storyline_building_support import get_rest_base_data, base_events


def get_baseline_storyline(save=False):
    rest_data = get_rest_base_data()
    for i in [5, 6, 7, 8]:
        rest_data[i] = 0
    data = pd.DataFrame(index=pd.date_range('2024-07-01', '2027-06-01', freq='MS'),
                        columns=['precip_class', 'temp_class', 'rest'])
    data.index.name = 'date'
    data.loc[:, 'year'] = data.index.year
    data.loc[:, 'month'] = data.index.month
    for i, y, m in data.loc[:, ['year', 'month']].itertuples(True, None):
        t, p, r = base_events[m]
        data.loc[i, 'precip_class'] = p
        data.loc[i, 'temp_class'] = t
        data.loc[i, 'rest'] = r

    data.loc[:, 'rest'] = data.loc[:, 'month'].replace(rest_data)
    ensure_no_impossible_events(data)
    if save:
        data.to_csv(os.path.join(climate_shocks_env.storyline_dir, '0-baseline.csv'))

    return data


if __name__ == '__main__':
    get_baseline_storyline(True)
