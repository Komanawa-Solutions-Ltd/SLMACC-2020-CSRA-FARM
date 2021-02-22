"""
 Author: Matt Hanson
 Created: 7/12/2020 10:58 AM
 """
import pandas as pd
import numpy as np
import os
import ksl_env
from Pasture_Growth_Modelling.initialisation_support.pasture_growth_deficit import calc_past_pasture_growth_anomaly
from Climate_Shocks.note_worthy_events.rough_stats import make_data
from Climate_Shocks.climate_shocks_env import event_def_dir, event_def_path

hot = '07d_d_tmax_25'
cold = '10d_d_tmean_07'
dry = '10d_d_smd000_sma-20'
monthly_smd_dry = '10d_d_smd000_sma-15'
wet = 'org'  # moved wet to n days with rain in month
rest = 'eqliklyd_rest'

events = [
    ('hot', hot),
    ('rolling_cold', cold),
    # ('dry', dry), this had some problems with reliance on previous month...
    ('monthly_smd_dry', monthly_smd_dry),  # trying in version 6 to see if it help the data matching
    ('ndays_wet', wet),
    ('rest', rest),

]

event_names = ['hot', 'cold', 'dry', 'wet', 'rest']
_org_describe_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
_describe_names = []
for e in _org_describe_names:
    _describe_names.extend(['{}_irr'.format(e), '{}_dry'.format(e)])

irrigated_pga = calc_past_pasture_growth_anomaly('irrigated', site='eyrewell').reset_index()

irrigated_pga.loc[:, 'year'] = irrigated_pga.date.dt.year
irrigated_pga = irrigated_pga.set_index(['month', 'year'])
dryland_pga = calc_past_pasture_growth_anomaly('dryland').reset_index()
dryland_pga.loc[:, 'year'] = dryland_pga.date.dt.year
dryland_pga = dryland_pga.set_index(['month', 'year'])


def add_pga(idx):
    idx = idx.dropna()
    irr_temp = irrigated_pga.loc[idx].reset_index()
    irr_temp2 = irr_temp.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:, 'pga_norm']
    dry_temp = dryland_pga.loc[idx].reset_index()
    dry_temp2 = dry_temp.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:, 'pga_norm']

    temp3 = pd.merge(irr_temp2, dry_temp2, left_index=True, right_index=True, suffixes=('_irr', '_dry'))
    return pd.DataFrame(temp3)


def make_prob(in_series):
    in_series = in_series.dropna()
    data = pd.DataFrame(np.atleast_2d(list(in_series.values)), columns=['month', 'year'])
    out_series = data.groupby('month').count() / 48
    return pd.DataFrame(out_series)


def get_org_data(event_dir=event_def_dir):
    data = [
        pd.read_csv(os.path.join(event_dir, '{}_years.csv'.format(f))).loc[:, k] for (f, k) in events

    ]
    use_data = []
    for d in data:
        use_data.append(
            pd.Series([np.nan if isinstance(t, float) else tuple(int(e) for e in t.strip('()').split(',')) for t in d]))

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
                           columns=pd.MultiIndex.from_product((full_event_names,
                                                               (['prob'] + _describe_names))
                                                              , names=['event', 'pga_desc']), dtype=float)
    org_data = get_org_data()
    # make base data
    print('making base data')
    for en in event_names:
        temp = make_prob(org_data.loc[:, en])
        outdata.loc[temp.index, (en, 'prob')] = temp.values[:, 0]
        temp = add_pga(org_data.loc[:, en])
        outdata.loc[temp.index, (en, _describe_names)] = temp.loc[:, _describe_names].values

    # make unique data
    print('making unique data')
    for en in event_names:
        en_u = '{}_unique'.format(en)
        idx = make_unique_idx(en, org_data)
        temp = make_prob(idx)
        outdata.loc[temp.index, (en_u, 'prob')] = temp.values[:, 0]
        temp = add_pga(idx)
        outdata.loc[temp.index, (en_u, _describe_names)] = temp.loc[:, _describe_names].values

    outdata = outdata.sort_index(axis=1, level=0, sort_remaining=False)
    return outdata


if __name__ == '__main__':
    # initial events recurrence must be run first. then this creates, the final events.
    # visualied_events.csv come from Storylines.check_storyline
    # event def data comes from Climate_Shocks\note_worthy_events\rough_stats.py
    run_old = False
    run_detrend_test = True
    if run_old:
        out = make_prob_impact_data()

        t = pd.Series([' '.join(e) for e in out.columns])
        idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
        out.loc[:, out.columns[idx]] *= 100
        out.to_csv(os.path.join(event_def_dir, 'current_choice.csv'), float_format='%.1f%%')
        out.to_csv(os.path.join(os.path.dirname(event_def_path),
                                'event_historical_prob_impact.csv'), float_format='%.1f%%')
        make_data(get_org_data(), save=True)
    if run_detrend_test:
        temp = make_data(get_org_data(ksl_env.shared_drives(r"Z2003_SLMACC\event_definition/v6_detrend")), save=True,
                         save_paths=[
                             ksl_env.shared_drives(r"Z2003_SLMACC\event_definition/v6_detrend/detrend_event_data.csv")])
        old = pd.read_csv(ksl_env.shared_drives(r"Z2003_SLMACC\event_definition\v5_detrend\detrend_event_data.csv"),
                          skiprows=1,
                          index_col=0, )
        temp.loc[:, 'old_temp'] = old.loc[:, 'temp'].values
        temp.loc[:, 'old_precip'] = old.loc[:, 'precip'].values
        temp.loc[:, 'change_temp'] = ~(temp.temp == temp.old_temp)
        temp.loc[:, 'change_precip'] = ~(temp.precip == temp.old_precip)
        temp.to_csv(ksl_env.shared_drives(r"Z2003_SLMACC\event_definition/v6_detrend/event_data_with_old.csv"))
        temp.loc[:, ['month', 'change_temp', 'change_precip']].groupby('month').sum().to_csv(
            ksl_env.shared_drives(r"Z2003_SLMACC\event_definition/v6_detrend/event_data_sum_changes.csv"))
