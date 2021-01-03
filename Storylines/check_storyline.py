"""
 Author: Matt Hanson
 Created: 24/12/2020 9:20 AM
 """
import pandas as pd
import numpy as np
from Climate_Shocks.climate_shocks_env import event_def_path
import itertools


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
    assert set(storyline.month) == set(np.arange(1, 13))
    # todo ensure that the months are reasonable and continious

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
    for k, v in get_acceptable_events().items():
        print(k)
        print(v)
        print('\n')
