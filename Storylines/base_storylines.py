"""
 Author: Matt Hanson
 Created: 27/01/2021 9:52 AM
 """
import pandas as pd


def get_baseline_storyline():
    data = pd.DataFrame(index=pd.date_range('2025-07-01', '2028-06-01', freq='MS'),
                        columns=['precip_class', 'temp_class', 'rest'])
    data.index.name = 'date'
    data.loc[:, 'year'] = data.index.year
    data.loc[:, 'year'] = data.index.month
    for i, y, m in data.loc[:, ['year', 'month']].itertuples(True, None):
        data.loc[i, 'precip_class'] = 'A'
        data.loc[i, 'temp_class'] = 'A'
        # todo what to do with restrictions data.loc[i, 'rest'] =, detrend the restriction data?

        #todo set the value to cold I think where a is not avalible