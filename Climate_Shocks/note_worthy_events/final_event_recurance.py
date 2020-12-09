"""
 Author: Matt Hanson
 Created: 7/12/2020 10:58 AM
 """
import pandas as pd
import numpy as np
import os
from Climate_Shocks.note_worthy_events.inital_event_recurance import backed_dir
from Pasture_Growth_Modelling.initialisation_support.pasture_growth_deficit import calc_past_pasture_growth_anomaly

hot = '07d_d_tmax_25'  # todo confirm
cold = '10d_d_tmean_07'  # todo confirm
dry = '10d_d_smd000_sma-20'  # todo confirm
wet = '10d_d_r0_smd-5'  # todo confirm
rest = '7d_half_7tot'  # todo confirm, update as we need to move back to number of days only

events = [
    ('hot', hot),
    ('rolling_cold', cold),
    ('dry', dry),
    ('smd_wet', wet),
    ('rest', rest),

]

event_names = ['hot', 'cold', 'dry', 'wet', 'rest']
desc_names = ['prob', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
_describe_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

irrigated_pga = calc_past_pasture_growth_anomaly('irrigated').reset_index()
irrigated_pga.loc[:, 'year'] = irrigated_pga.date.dt.year
irrigated_pga = irrigated_pga.set_index(['month', 'year'])


def add_pga(idx):
    idx = idx.dropna()
    temp = irrigated_pga.loc[idx].reset_index()
    temp2 = temp.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:, 'pga_norm']
    return pd.DataFrame(temp2).round(3)


def make_prob(in_series):
    in_series = in_series.dropna()
    data = pd.DataFrame(np.atleast_2d(list(in_series.values)), columns=['month', 'year'])
    out_series = data.groupby('month').count() / 48
    return pd.DataFrame(out_series).round(2)


def get_org_data():
    data = [
        pd.read_csv(os.path.join(backed_dir, '{}_years.csv'.format(f))).loc[:, k] for (f, k) in events

    ]
    use_data = []
    for d in data:
        use_data.append(pd.Series([np.nan if isinstance(t,float) else tuple(int(e) for e in t.strip('()').split(',')) for t in d]))

    data = pd.concat(use_data, axis=1)
    data.columns = event_names

    return data


def make_unique_idx(k, org_data):
    out = set(org_data.loc[:, k])
    for en in event_names:
        if en == k:
            continue
        out = out - set(org_data.loc[:, en])

    return pd.Series(list(out))


def make_prob_impact_data():
    full_event_names = event_names + ['{}_unique'.format(e) for e in event_names]

    outdata = pd.DataFrame(index=pd.Series(range(1, 13), name='month'),
                           columns=pd.MultiIndex.from_product((full_event_names, desc_names), names=['event',
                                                                                                     'pga_desc']))
    org_data = get_org_data()
    # make base data
    print('making base data')
    for en in event_names:
        temp = make_prob(org_data.loc[:, en])
        outdata.loc[temp.index, (en, 'prob')] = temp.values[:,0]
        temp =  add_pga(org_data.loc[:, en])
        outdata.loc[temp.index, (en, _describe_names)] = temp.values

    # make unique data
    print('making unique data')
    for en in event_names:
        en_u = '{}_unique'.format(en)
        idx = make_unique_idx(en, org_data)
        temp = make_prob(idx)
        outdata.loc[temp.index, (en_u, 'prob')] = temp.values[:,0]
        temp =  add_pga(idx)
        outdata.loc[temp.index, (en_u, _describe_names)] = temp.values

    outdata = outdata.sort_index(axis=1, level=0, sort_remaining=False)
    return outdata


if __name__ == '__main__':
    # todo check and then finalize events
    test = make_prob_impact_data()
    test.to_csv(os.path.join(backed_dir, 'current_choice.csv'))
