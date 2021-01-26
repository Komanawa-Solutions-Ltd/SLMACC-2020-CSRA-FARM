"""
 Author: Matt Hanson
 Created: 27/01/2021 9:52 AM
 """
import pandas as pd
from Storylines.check_storyline import ensure_no_impossible_events

def get_baseline_storyline():
    data = pd.DataFrame(index=pd.date_range('2025-07-01', '2028-06-01', freq='MS'),
                        columns=['precip_class', 'temp_class', 'rest'])
    data.index.name = 'date'
    data.loc[:, 'year'] = data.index.year
    data.loc[:, 'month'] = data.index.month
    for i, y, m in data.loc[:, ['year', 'month']].itertuples(True, None):
        data.loc[i, 'precip_class'] = 'A'
        data.loc[i, 'temp_class'] = 'A'
        data.loc[i, 'rest'] = 0 # todo what to do with restrictions detrend the restriction data and use mean?
        if m == 7:
            data.loc[i, 'temp_class'] = 'C' # no average temp julys

    ensure_no_impossible_events(data)
    return data

if __name__ == '__main__':
    get_baseline_storyline()