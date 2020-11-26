"""
 Author: Matt Hanson
 Created: 3/11/2020 9:04 AM
 """

from Climate_Shocks.vcsn_pull import vcsn_pull_single_site
from Climate_Shocks.note_worthy_events.simple_smd_soilt import calc_sma_smd
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
import ksl_env
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

backed_dir = ksl_env.shared_drives("SLMACC_2020\event_definition")
unbacked_dir = ksl_env.mh_unbacked("SLMACC_2020\event_definition")

if not os.path.exists(backed_dir):
    os.makedirs(backed_dir)

if not os.path.exists(unbacked_dir):
    os.makedirs(unbacked_dir)


def prob(x):
    out = np.nansum(x) / len(x)
    return np.round(out, 2)


def calc_dry_recurance():
    data = get_vcsn_record()

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
    out.to_csv(os.path.join(backed_dir, 'dry_prob.csv'))


def calc_wet_recurance():
    data = get_vcsn_record()
    data.to_csv(os.path.join(backed_dir, 'wet_raw.csv'))

    thresholds = [15, 10, 7, 5, 3, 1]
    ndays = [1, 5, 7, 10, 14]
    out_keys = []
    for thresh in thresholds:
        k = 'd_rain_cond_{:02d}'.format(thresh)
        data.loc[:, k] = data.loc[:, 'rain'] >= thresh
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

    grouped_data.to_csv(os.path.join(backed_dir, 'wet_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(backed_dir,
                                                                                        'wet_monthly_data_desc.csv'))

    # number of n days
    out_keys2 = []
    for nd in ndays:
        for t, k in zip(thresholds, out_keys):
            ok = '{:02d}d r{:02d}'.format(nd, t)
            out_keys2.append(ok)
            grouped_data.loc[:, ok] = grouped_data.loc[:, k] >= nd

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    out.to_csv(os.path.join(backed_dir, 'wet_prob.csv'))


def calc_restrict_recurance():
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
    out.to_csv(os.path.join(backed_dir, 'rest_prob.csv'))


def calc_cold_recurance():
    data = get_vcsn_record()
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2
    data.to_csv(os.path.join(backed_dir, 'temp_raw.csv'))

    thresholds = [0, 5, 7, 10, 12]
    vars = ['tmin', 'tmean']
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

    grouped_data.to_csv(os.path.join(backed_dir, 'cold_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(backed_dir,
                                                                                        'cold_monthly_data_desc.csv'))

    # number of n days
    out_keys2 = []
    for nd in ndays:
        for k in out_keys:
            ok = '{:02d}d_{}'.format(nd, k)
            out_keys2.append(ok)
            grouped_data.loc[:, ok] = grouped_data.loc[:, k] >= nd

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    out.to_csv(os.path.join(backed_dir, 'cold_prob.csv'))


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
    out.to_csv(os.path.join(backed_dir, 'hot_prob.csv'))


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


if __name__ == '__main__':
    calc_hot_recurance()
    calc_cold_recurance()
    calc_restrict_recurance()
    calc_wet_recurance()
    calc_dry_recurance()
