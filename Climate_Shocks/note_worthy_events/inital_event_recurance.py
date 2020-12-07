"""
 Author: Matt Hanson
 Created: 3/11/2020 9:04 AM
 """

from Climate_Shocks.vcsn_pull import vcsn_pull_single_site
from Climate_Shocks.note_worthy_events.simple_smd_soilt import calc_sma_smd
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Pasture_Growth_Modelling.initialisation_support.pasture_growth_deficit import calc_past_pasture_growth_anomaly
import ksl_env
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools

backed_dir = ksl_env.shared_drives("SLMACC_2020\event_definition/v2")
unbacked_dir = ksl_env.mh_unbacked("SLMACC_2020\event_definition")

if not os.path.exists(backed_dir):
    os.makedirs(backed_dir)

if not os.path.exists(unbacked_dir):
    os.makedirs(unbacked_dir)

irrigated_pga = calc_past_pasture_growth_anomaly('irrigated').reset_index()
irrigated_pga.loc[:, 'year'] = irrigated_pga.date.dt.year
irrigated_pga = irrigated_pga.set_index(['month', 'year'])


def prob(x):
    out = np.nansum(x) / len(x)
    return np.round(out, 2)


def add_pga(grouped_data, sim_keys, outdata):
    grouped_data = grouped_data.set_index(['month', 'year'])
    years = {}
    for k in sim_keys:
        idx = grouped_data.loc[grouped_data.loc[:, k], k]
        assert idx.all()
        idx = idx.index
        years[k] = idx.values
        temp = irrigated_pga.loc[idx].reset_index()
        temp2 = temp.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:,'pga_norm']
        for k2 in temp2:
            outdata.loc[:, (k, 'pga_{}'.format(k2))] = temp2.loc[:, k2]

    mx_years = 48*12 + 1
    out_years = pd.DataFrame(index=range(mx_years), columns=sim_keys)
    for k in sim_keys:
        missing_len = mx_years - len(years[k])
        out_years.loc[:, k] = np.concatenate((years[k], np.zeros(missing_len) * np.nan))

    outdata = outdata.sort_index(axis=1, level=0, sort_remaining=False)

    return outdata, out_years


def calc_dry_recurance():
    data = get_vcsn_record().reset_index()

    temp = calc_sma_smd(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]
    data.to_csv(os.path.join(backed_dir, 'dry_raw.csv'))

    smd_thresholds = [0, -110, -110]
    sma_thresholds = [-20, 0, -20]
    ndays = [5, 7, 10, 14]
    out_keys = []
    for smd_t, sma_t in zip(smd_thresholds, sma_thresholds):
        k = 'd_smd{:03d}_sma{:02d}'.format(smd_t, sma_t)
        data.loc[:, k] = (data.loc[:, 'smd'] <= smd_t) & (data.loc[:, 'sma'] <= sma_t)
        out_keys.append(k)

    grouped_data = data.loc[:, ['month', 'year',
                                'smd', 'sma'] + out_keys].groupby(['month', 'year']).sum().reset_index()

    grouped_data.to_csv(os.path.join(backed_dir, 'dry_monthly_data.csv'))
    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(backed_dir,
                                                                                        'dry_monthly_data_desc.csv'))
    out_keys2 = []
    for nd in ndays:
        for k in out_keys:
            ok = '{:02d}d_{}'.format(nd, k)
            out_keys2.append(ok)
            grouped_data.loc[:, ok] = grouped_data.loc[:, k] >= nd

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    out.to_csv(os.path.join(backed_dir, 'dry_prob.csv'))
    out_years.to_csv(os.path.join(backed_dir, 'dry_years.csv'))


def calc_wet_recurance():
    data = get_vcsn_record().reset_index()
    temp = calc_sma_smd(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]

    temp = False
    if temp:  # just to look at some plots
        fig, (ax, ax2, ax3) = plt.subplots(3, sharex=True)
        ax.plot(data.date, data.smd)
        ax2.plot(data.date, data.drain)
        ax3.plot(data.date, data.rain)
        plt.show()

    data.to_csv(os.path.join(backed_dir, 'smd_wet_raw.csv'))

    thresholds_rain = [5, 3, 1, 0]
    thresholds_smd = [0, -5, -10]
    ndays = [7, 10, 14]
    out_keys = []
    for t_r, t_smd in itertools.product(thresholds_rain, thresholds_smd):
        k = 'd_r{}_smd{}'.format(t_r, t_smd)
        data.loc[:, k] = (data.loc[:, 'rain'] >= t_r) & (data.loc[:, 'smd'] >= t_smd)
        out_keys.append(k)

    grouped_data = data.loc[:, ['month', 'year', 'rain'] + out_keys].groupby(['month', 'year']).sum().reset_index()

    # make montly restriction anaomaloy - mean
    temp = grouped_data.groupby('month').mean().loc[:, 'rain'].to_dict()
    grouped_data.loc[:, 'rain_an_mean'] = grouped_data.loc[:, 'month']
    grouped_data = grouped_data.replace({'rain_an_mean': temp})
    grouped_data.loc[:, 'rain_an_mean'] = grouped_data.loc[:, 'rain'] - grouped_data.loc[:, 'rain_an_mean']

    # make montly restriction anaomaloy - median
    temp = grouped_data.groupby('month').median().loc[:, 'rain'].to_dict()
    grouped_data.loc[:, 'rain_an_med'] = grouped_data.loc[:, 'month']
    grouped_data = grouped_data.replace({'rain_an_med': temp})
    grouped_data.loc[:, 'rain_an_med'] = grouped_data.loc[:, 'rain'] - grouped_data.loc[:, 'rain_an_med']

    grouped_data.round(2).to_csv(os.path.join(backed_dir, 'smd_wet_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().round(2).to_csv(os.path.join(backed_dir,
                                                                                                 'smd_wet_monthly_data_desc.csv'))

    # number of n days
    out_keys2 = []
    for nd in ndays:
        for k in out_keys:
            ok = '{:02d}d_{}'.format(nd, k)
            out_keys2.append(ok)
            grouped_data.loc[:, ok] = grouped_data.loc[:, k] >= nd

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob]).round(2)
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    out.to_csv(os.path.join(backed_dir, 'smd_wet_prob.csv'))
    out_years.to_csv(os.path.join(backed_dir, 'smd_wet_years.csv'))


def old_calc_restrict_recurance():
    data = get_restriction_record()

    thresholds = [0.001, 0.5, 0.75, 1]
    tnames = ['any', 'half', '3/4', 'full']
    ndays = [1, 5, 7, 10, 14]
    out_keys = []
    for thresh, tname in zip(thresholds, tnames):
        k = 'd_>{}_rest'.format(tname)
        data.loc[:, k] = data.loc[:, 'f_rest'] >= thresh
        out_keys.append(k)

    grouped_data = data.loc[:, ['month', 'year', 'f_rest'] + out_keys].groupby(['month', 'year']).sum().reset_index()

    # make montly restriction anaomaloy - mean
    temp = grouped_data.groupby('month').mean().loc[:, 'f_rest'].to_dict()
    grouped_data.loc[:, 'f_rest_an_mean'] = grouped_data.loc[:, 'month']
    grouped_data = grouped_data.replace({'f_rest_an_mean': temp})
    grouped_data.loc[:, 'f_rest_an_mean'] = grouped_data.loc[:, 'f_rest'] - grouped_data.loc[:, 'f_rest_an_mean']

    # make montly restriction anaomaloy
    temp = grouped_data.groupby('month').median().loc[:, 'f_rest'].to_dict()
    grouped_data.loc[:, 'f_rest_an_med'] = grouped_data.loc[:, 'month']
    grouped_data = grouped_data.replace({'f_rest_an_med': temp})
    grouped_data.loc[:, 'f_rest_an_med'] = grouped_data.loc[:, 'f_rest'] - grouped_data.loc[:, 'f_rest_an_med']

    grouped_data.to_csv(os.path.join(backed_dir, 'rest_monthly_data.csv'))
    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(backed_dir,
                                                                                        'rest_monthly_data_desc.csv'))
    # number of n days
    out_keys2 = []
    for nd in ndays:
        for k in out_keys:
            ok = '{:02d}d_{}'.format(nd, k)
            out_keys2.append(ok)
            grouped_data.loc[:, ok] = grouped_data.loc[:, k] >= nd

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (
                out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out.to_csv(os.path.join(backed_dir, 'rest_prob.csv')) #todo pull this back up/in


def calc_restrict_recurance():
    data = get_restriction_record()

    thresholds = [0.001, 0.5, 0.75, 1]
    tnames = ['any', 'half', '75rest', 'full']
    con_days = [5, 7, 10]
    ndays = [5, 7, 10, 15, 20]

    consecutive_data = {}
    for tnm, t in zip(tnames, thresholds):
        test_value = tnm
        data.loc[:, test_value] = data.loc[:, 'f_rest'] >= t

        data.loc[:, 'con_id'] = (data.loc[:, ['year',
                                              'month',
                                              test_value]].diff(1) != 0).any(axis=1).astype('int').cumsum().values

        temp = data.loc[data[test_value]].groupby('con_id')
        consecutive_data[tnm] = temp.agg({'year': 'mean', 'month': 'mean', test_value: 'size'}).reset_index()

    out_columns = ['total_rest_days', 'num_per', 'mean_per_len', 'min_per_len', 'max_per_len']
    rename_mapper = {'sum': 'total_rest_days', 'count': 'num_per',
                     'mean': 'mean_per_len', 'min': 'min_per_len', 'max': 'max_per_len'}

    all_data = pd.DataFrame(
        index=pd.MultiIndex.from_product([set(data.year), set(data.month)], names=['year', 'month']),
        columns=pd.MultiIndex.from_product([tnames, out_columns]))
    all_data.loc[:] = np.nan
    for k, v in consecutive_data.items():
        v.round(2).to_csv(os.path.join(backed_dir, 'len_rest_{}_raw.csv'.format(k)))
        temp = v.groupby(['year', 'month']).agg({k: ['sum', 'count',
                                                     'mean', 'min', 'max']})
        temp = temp.rename(columns=rename_mapper, level=1)
        all_data = all_data.combine_first(temp)

    all_data = all_data.loc[:, (tnames, out_columns)]
    all_data.reset_index().astype(float).groupby('month').describe().round(2).to_csv(os.path.join(backed_dir,
                                                                                                  'len_rest_month_desc_no_zeros.csv'))
    t = all_data['any']['num_per'].isna().reset_index().groupby('month').agg({'num_per': ['sum', prob]})
    t.to_csv(os.path.join(backed_dir, 'len_rest_prob_no_rest.csv'))
    all_data = all_data.fillna(0)
    all_data.round(2).to_csv(os.path.join(backed_dir, 'len_rest_monthly.csv'))

    all_data.reset_index().groupby('month').describe().round(2).to_csv(
        os.path.join(backed_dir, 'len_rest_month_desc_with_zeros.csv'))

    prob_data = pd.DataFrame(index=all_data.index)

    for rt, l, nd in itertools.product(tnames, con_days, ndays):
        prob_data.loc[:, '{}d_{}_{}tot'.format(l, rt, nd)] = ((all_data.loc[:, (rt, 'max_per_len')] >= l) &
                                                              (all_data.loc[:, (rt, 'total_rest_days')] >= nd))

    out = prob_data.reset_index().groupby('month').agg(['sum', prob])
    out_keys2 = set(out.columns.levels[0]) - {'year'}
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (
                out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(prob_data.reset_index(), set(out_keys2) - set(drop_keys), out)
    out.to_csv(os.path.join(backed_dir, 'len_rest_prob.csv'))
    out_years.to_csv(os.path.join(backed_dir, 'len_rest_years.csv'))


def calc_cold_recurance():
    data = get_vcsn_record()
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2
    data.loc[:, 'tmean_raw'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2
    data.loc[:, 'tmean'] = data.loc[:, 'tmean'].rolling(3).mean()
    data.round(2).to_csv(os.path.join(backed_dir, 'rolling_cold_raw.csv'))

    thresholds = [0, 5, 7, 10, 12]
    vars = ['tmean']
    ndays = [3, 5, 7, 10, 14]
    out_keys = []
    for thresh, v in itertools.product(thresholds, vars):
        k = 'd_{}_{:02d}'.format(v, thresh)
        data.loc[:, k] = data.loc[:, v] <= thresh
        out_keys.append(k)

    aggs = {e: 'sum' for e in out_keys}
    aggs.update({e: 'mean' for e in vars})
    grouped_data = data.loc[:, ['month', 'year'] + vars + out_keys].groupby(['month', 'year'])
    grouped_data = grouped_data.aggregate(aggs).reset_index()

    grouped_data.round(2).to_csv(os.path.join(backed_dir, 'rolling_cold_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().round(2).to_csv(os.path.join(backed_dir,
                                                                                                 'rolling_cold_monthly_data_desc.csv'))

    # number of n days
    out_keys2 = []
    for nd in ndays:
        for k in out_keys:
            ok = '{:02d}d_{}'.format(nd, k)
            out_keys2.append(ok)
            grouped_data.loc[:, ok] = grouped_data.loc[:, k] >= nd

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])

    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (
                out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    out.to_csv(os.path.join(backed_dir, 'rolling_cold_prob.csv'))
    out_years.to_csv(os.path.join(backed_dir, 'rolling_cold_years.csv'))


def calc_hot_recurance():
    data = get_vcsn_record()
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2
    data.to_csv(os.path.join(backed_dir, 'temp_raw.csv'))

    thresholds = [20, 25, 28, 30, 35]
    vars = ['tmax', 'tmean']
    ndays = [3, 5, 7, 10, 14]
    out_keys = []
    for thresh, v in itertools.product(thresholds, vars):
        k = 'd_{}_{:02d}'.format(v, thresh)
        data.loc[:, k] = data.loc[:, v] >= thresh
        out_keys.append(k)

    aggs = {e: 'sum' for e in out_keys}
    aggs.update({e: 'mean' for e in vars})
    grouped_data = data.loc[:, ['month', 'year'] + vars + out_keys].groupby(['month', 'year'])
    grouped_data = grouped_data.aggregate(aggs).reset_index()

    grouped_data.to_csv(os.path.join(backed_dir, 'hot_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(backed_dir,
                                                                                        'hot_monthly_data_desc.csv'))

    # number of n days
    out_keys2 = []
    for nd in ndays:
        for k in out_keys:
            ok = '{:02d}d_{}'.format(nd, k)
            out_keys2.append(ok)
            grouped_data.loc[:, ok] = grouped_data.loc[:, k] >= nd

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (
                out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    out.to_csv(os.path.join(backed_dir, 'hot_prob.csv'))
    out_years.to_csv(os.path.join(backed_dir, 'hot_years.csv'))


def plot_vcsn_smd():
    data, use_cords = vcsn_pull_single_site(
        lat=-43.358,
        lon=172.301,
        year_min=1972,
        year_max=2019,
        use_vars=('evspsblpot', 'pr'))
    print(use_cords)

    temp = calc_sma_smd(data['pr'], data['evspsblpot'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]

    data.set_index('date', inplace=True)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.plot(data.index, data['evspsblpot'], label='pet')
    ax1.plot(data.index, data['aet_out'], label='aet')
    ax2.plot(data.index, data['pr'], label='rain')
    ax3.plot(data.index, data['smd'], label='smd')
    ax3.plot(data.index, data['mean_doy_smd'], label='daily_mean_smd')
    ax4.plot(data.index, data['sma'], label='sma')
    ax4.axhline(ls='--', c='k')

    for ax in (ax1, ax2, ax3, ax4):
        ax.legend()

    plt.show()


def check_vcns_data():
    data, use_cords = vcsn_pull_single_site(
        lat=-43.358,
        lon=172.301,
        year_min=1972,
        year_max=2019,
        use_vars='all')

    print(use_cords)
    data.set_index('date', inplace=True)
    for v in data.keys():
        fix, (ax) = plt.subplots()
        ax.plot(data.index, data[v])
        ax.set_title(v)
    plt.show()


def plot_restriction_record():
    data = get_restriction_record()
    fix, (ax) = plt.subplots()
    ax.plot(pd.to_datetime(data['date']), data['f_rest'])
    plt.show()


if __name__ == '__main__':
    calc_restrict_recurance()
    calc_dry_recurance()
    calc_wet_recurance()
    calc_cold_recurance()

    calc_hot_recurance()
