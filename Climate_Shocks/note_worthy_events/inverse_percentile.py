"""
 Author: Matt Hanson
 Created: 11/01/2021 1:35 PM
 """
import numpy as np
import pandas as pd
import os
from scipy import stats
from Climate_Shocks.get_past_record import get_vcsn_record
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_sma_smd_historical
from Storylines.check_storyline import get_months_with_events
from Climate_Shocks.climate_shocks_env import event_def_path


def inverse_percentile(a, value, bootstrap=True):
    """
    calculate the percentile (per) represented in array by the value returns the error of the calculations as well
    :param a: np.array
    :param value: value to calculate the percentile of
    :return: per, err
    """
    if bootstrap:
        a = np.random.choice(a, 10000)
    per = stats.percentileofscore(a, value)
    # then check
    v = np.percentile(a, per)
    err = np.abs(value - v)
    return per, err


def calc_doy_per_from_historical(version='trended'):
    data = get_vcsn_record(version).reset_index()
    data.loc[:, 'doy'] = data.date.dt.dayofyear
    data.loc[:, 'cold'] = ((data.loc[:, 'tmin'] + data.loc[:, 'tmax']) / 2).rolling(3).mean()
    data.loc[:, 'hot'] = data.loc[:, 'tmax']
    data.loc[:, 'wet'] = data.loc[:, 'rain']
    temp = calc_sma_smd_historical(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]
    data.loc[:, 'dry'] = data.loc[:, 'sma']
    use_keys = ['hot', 'cold', 'dry', 'wet']
    thresholds = {
        'hot': 25,
        'cold': 7,
        'dry': -20,
        'wet': 0.01,
    }
    use_keys2 = ['H', 'C', 'D', 'W']
    events = get_months_with_events()
    outdata = pd.DataFrame(index=pd.Index(range(1, 366), name='dayofyear'))
    for k, k2 in zip(use_keys, use_keys2):
        print(k)
        temp = data.loc[np.in1d(data.month, events[k2])]
        for d in range(1, 366):
            temp2 = temp.loc[temp.doy == d, k]
            if temp2.empty:
                continue
            per, err = inverse_percentile(temp2, thresholds[k], bootstrap=False)
            outdata.loc[d, '{}_per'.format(k)] = per
            outdata.loc[d, '{}_err'.format(k)] = err
    outdata.loc[:, 'date'] = pd.to_datetime(['2001-{:03d}'.format(e) for e in outdata.index], format='%Y-%j')
    outdata.loc[:,'month'] = outdata.date.dt.month

    # get rid of hangers on from leap years
    for k, k2 in zip(use_keys, use_keys2):
        outdata.loc[~np.in1d(outdata.month, events[k2]), '{}_per'.format(k)] = np.nan
        outdata.loc[~np.in1d(outdata.month, events[k2]), '{}_err'.format(k)] = np.nan

    outdata.set_index('date', inplace=True, append=True)

    return outdata


if __name__ == '__main__':
    data = calc_doy_per_from_historical('detrended')
    data.to_csv(os.path.join(os.path.dirname(event_def_path), 'daily_percentiles_detrended.csv')) #todo debug, run, and check.
    data = calc_doy_per_from_historical()
    data.to_csv(os.path.join(os.path.dirname(event_def_path), 'daily_percentiles.csv'))
