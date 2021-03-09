"""
 Author: Matt Hanson
 Created: 24/12/2020 9:20 AM
 """
import pandas as pd
import numpy as np
import os
from Climate_Shocks.climate_shocks_env import event_def_path
import itertools
import ksl_env
import glob


def get_months_with_events():
    events = pd.read_csv(event_def_path, skiprows=1)
    temps = ['C', 'AT', 'H']
    precips = ['W', 'AP', 'D']
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


def get_past_event_frequency():
    events = pd.read_csv(event_def_path, skiprows=1)
    temps = ['C', 'AT', 'H']
    precips = ['W', 'AP', 'D']
    _vals = [-1, 0, 1]
    events.loc[:, 'precip'] = events.loc[:, 'precip'].replace({k: v for k, v in zip(_vals, precips)})
    events.loc[:, 'temp'] = events.loc[:, 'temp'].replace({k: v for k, v in zip(_vals, temps)})
    events.loc[:, 'state'] = ['{}-{}'.format(t, p) for t, p in zip(events.temp, events.precip)]
    events = events.groupby(['month', 'state']).count()
    return events


def ensure_no_impossible_events(storyline): # takes longer to run this than the IID
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
                                  [ 6, 7, 8]), 'rest'] == 0).all(), 'irrigation rest in months without irr'

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
    # out_zero, out_not_zero, missing_data = get_all_zero_prob_transitions()
    # 'month:{} to {}'.format(m, m2)
    # '{} to {} is zero'.format(state1, state2)
    if problems:
        raise ValueError('\n '.join(messages))


def get_all_zero_prob_transitions(save=True):
    trans_prob_dir = os.path.join(ksl_env.proj_root, 'BS_work/IID/TransitionProbabilities')
    trans = _read_trans(glob.glob(os.path.join(trans_prob_dir, '*_transitions.csv')))
    events = {e: [] for e in range(1, 13)}
    acceptable = get_acceptable_events()
    for k, v in acceptable.items():
        for m in v:
            t, p = k.split('-')
            events[m].append('{}-{}'.format(t, p))
    out_zero = {}
    out_not_zero = {}
    missing_data = {}
    for m in range(1, 13):
        if m == 12:
            m2 = 1
        else:
            m2 = m + 1
        out_zero['month:{} to {}'.format(m, m2)] = []
        out_not_zero['month:{} to {}'.format(m, m2)] = []
        missing_data['month:{} to {}'.format(m, m2)] = []
        current_m = events[m]
        next_m = events[m2]

        for state1, state2 in itertools.product(current_m, next_m):
            if trans[m].loc[state2, state1] == 0:
                out_zero['month:{} to {}'.format(m, m2)].append('{} to {} is zero'.format(state1, state2))
            else:
                out_not_zero['month:{} to {}'.format(m, m2)].append('{} to {} is NOT zero'.format(state1, state2))
            if (trans[m].loc[:, state1] == 0).all():
                missing_data['month:{} to {}'.format(m, m2)].append(state1)

    if save:
        with open(os.path.join(os.path.dirname(event_def_path), 'not_permitted_trans.txt'),'w') as f:
            for k, v in out_zero.items():
                f.write('{}: {}\n'.format(k,v))
        with open(os.path.join(os.path.dirname(event_def_path), 'permitted_trans.txt'),'w') as f:
            for k, v in out_not_zero.items():
                f.write('{}: {}\n'.format(k,v))

    return out_zero, out_not_zero, missing_data


def _read_trans(paths):
    out = {}
    for path in paths:
        month = months[os.path.basename(path).split('_')[0].lower()]
        t = pd.read_csv(path,
                        skiprows=7, index_col=0)
        t.index = t.index.str.replace(',', '-')
        t.columns = t.columns.str.replace(',', '-')
        out[month] = t
    return out


months = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
}

if __name__ == '__main__':
    test = get_acceptable_events()
    temp = get_past_event_frequency()
    get_all_zero_prob_transitions()
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
