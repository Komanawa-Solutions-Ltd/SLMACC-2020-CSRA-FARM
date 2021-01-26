"""
 Author: Matt Hanson
 Created: 24/12/2020 9:20 AM
 """
import pandas as pd
import numpy as np
import os
from Climate_Shocks.climate_shocks_env import event_def_path
import itertools


def get_months_with_events():
    events = pd.read_csv(event_def_path, skiprows=1)
    temps = ['C', 'tA', 'H']
    precips = ['W', 'pA', 'D']
    _vals = [-1, 0, 1]
    events_out = {}
    for (tkey, tval) in zip(temps, _vals):
        temp = events.loc[np.isclose(events.temp, tval)].month.unique()
        events_out['{}'.format(tkey)] = temp
    for (pkey, pval) in zip(precips, _vals):
        temp = events.loc[np.isclose(events.precip, pval)].month.unique()
        events_out['{}'.format(pkey)] = temp

    return events_out


def get_acceptable_events():
    events = pd.read_csv(event_def_path, skiprows=1)
    temps = ['C', 'A', 'H']
    precips = ['W', 'A', 'D']
    _vals = [-1, 0, 1]
    acceptable_events = {}
    for (tkey, tval), (pkey, pval) in itertools.product(zip(temps, _vals), zip(precips, _vals)):
        temp = events.loc[np.isclose(events.temp, tval) & np.isclose(events.precip, pval)].month.unique()
        acceptable_events['{}-{}'.format(tkey, pkey)] = temp

    return acceptable_events


def ensure_no_impossible_events(storyline):
    assert isinstance(storyline, pd.DataFrame)
    assert set(storyline.columns) == {'year', 'month', 'temp_class', 'precip_class', 'rest'}
    assert set(storyline.temp_class.unique()).issubset(['C', 'A', 'H']), 'unexpected classes for temp_class'
    assert set(storyline.precip_class.unique()).issubset(['W', 'A', 'D']), 'unexpected classes for precip_class'
    assert storyline.rest.max() <= 1, 'unexpected values for restrictions'
    assert storyline.rest.min() >= 0, 'unexpected values for restrictions'

    # check dates
    assert set(storyline.month) == set(np.arange(1, 13))
    assert storyline.month.iloc[0] == 7, 'all storylines must start in july'
    years = storyline.year.unique()
    expected_dates = pd.date_range('{}-07-01'.format(min(years)), '{}-06-01'.format(max(years)), freq='MS')
    assert isinstance(storyline.index, pd.DatetimeIndex), 'storyline index must be datetime index'
    idx = storyline.index == expected_dates
    assert idx.all(), 'expected full years bad values are {}'.format(storyline.index[idx])
    # check against daterange for index.
    assert (storyline.loc[np.in1d(storyline.month,
                                  [5, 6, 7, 8]), 'rest'] == 0).all(), 'irrigation rest in months without irr'

    acceptable_events = get_acceptable_events()
    problems = False
    messages = []
    for k in storyline.index:
        month = storyline.loc[k, 'month']
        year = storyline.loc[k, 'year']
        combo_key = '{}-{}'.format(storyline.loc[k, 'temp_class'], storyline.loc[k, 'precip_class'])
        if month not in acceptable_events[combo_key]:
            messages.append('unacceptable combination(s):{} in year: {} month: {}'.format(combo_key, year, month))
            problems = True
    if problems:
        raise ValueError('\n '.join(messages))


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(event_def_path), 'visualize_event_options.csv')
    acceptable = get_acceptable_events()
    out_data = pd.DataFrame(index=pd.Index(range(1, 13), name='month'))
    for k, v in acceptable.items():
        k = k.replace('C', 'Cold').replace('A', 'Average').replace('H', 'Hot')
        k = k.replace('W', 'Wet').replace('D', 'Dry')
        out_data.loc[np.in1d(out_data.index, v), k] = True
        print(k)
        print(v)
        print('\n')
    print(out_data)
    out_data.to_csv(data_path)
