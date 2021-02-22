"""
 Author: Matt Hanson
 Created: 3/11/2020 9:04 AM
 """

from Climate_Shocks.vcsn_pull import vcsn_pull_single_site
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_sma_smd_historical, calc_smd_monthly
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Pasture_Growth_Modelling.initialisation_support.pasture_growth_deficit import calc_past_pasture_growth_anomaly
import ksl_env
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools
from Climate_Shocks.climate_shocks_env import event_def_dir

unbacked_dir = ksl_env.mh_unbacked("Z2003_SLMACC\event_definition")
if not os.path.exists(unbacked_dir):
    os.makedirs(unbacked_dir)

vcsn_version = 'trended'

# todo testing detrended data changes
test_detrended = True
if test_detrended:
    event_def_dir = ksl_env.shared_drives("Z2003_SLMACC\event_definition/v6_detrend")
    vcsn_version = 'detrended2'

if not os.path.exists(event_def_dir):
    os.makedirs(event_def_dir)

irrigated_pga = calc_past_pasture_growth_anomaly('irrigated', site='eyrewell').reset_index()
irrigated_pga.loc[:, 'year'] = irrigated_pga.date.dt.year
irrigated_pga = irrigated_pga.set_index(['month', 'year'])
dryland_pga = calc_past_pasture_growth_anomaly('dryland').reset_index()
dryland_pga.loc[:, 'year'] = dryland_pga.date.dt.year
dryland_pga = dryland_pga.set_index(['month', 'year'])


def prob(x):
    out = np.nansum(x) / len(x)
    return out


def add_pga_from_idx(idx):
    idx = idx.dropna()
    irr_temp = irrigated_pga.loc[idx].reset_index()
    irr_temp2 = irr_temp.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:, 'pga_norm']
    dry_temp = dryland_pga.loc[idx].reset_index()
    dry_temp2 = dry_temp.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:, 'pga_norm']

    temp3 = pd.merge(irr_temp2, dry_temp2, left_index=True, right_index=True, suffixes=('_irr', '_dry'))
    return pd.DataFrame(temp3)


def add_pga(grouped_data, sim_keys, outdata):
    grouped_data = grouped_data.set_index(['month', 'year'])
    years = {}
    for k in sim_keys:
        idx = grouped_data.loc[grouped_data.loc[:, k], k]
        assert idx.all()
        idx = idx.index
        years[k] = idx.values
        temp_irr = irrigated_pga.loc[idx].reset_index()
        temp_irr2 = temp_irr.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:, 'pga_norm']
        temp_dry = dryland_pga.loc[idx].reset_index()
        temp_dry2 = temp_dry.loc[:, ['month', 'pga_norm']].groupby('month').describe().loc[:, 'pga_norm']
        for k2 in temp_irr2:
            outdata.loc[:, (k, 'pga_irr_{}'.format(k2))] = temp_irr2.loc[:, k2]
            outdata.loc[:, (k, 'pga_dry_{}'.format(k2))] = temp_dry2.loc[:, k2]

    mx_years = 48 * 12 + 1
    out_years = pd.DataFrame(index=range(mx_years), columns=sim_keys)
    for k in sim_keys:
        missing_len = mx_years - len(years[k])
        out_years.loc[:, k] = np.concatenate((years[k], np.zeros(missing_len) * np.nan))

    outdata = outdata.sort_index(axis=1, level=0, sort_remaining=False)

    return outdata, out_years


def calc_dry_recurance():
    data = get_vcsn_record(vcsn_version)
    t = calc_smd_monthly(rain=data.rain, pet=data.pet, dates=data.index)
    data.loc[:, 'smd'] = t
    t = data.loc[:, ['doy', 'smd']].groupby('doy').mean().to_dict()
    data.loc[:, 'sma'] = data.loc[:, 'smd'] - data.loc[:, 'doy'].replace(t['smd'])
    data.reset_index(inplace=True)

    data.to_csv(os.path.join(event_def_dir, 'monthly_smd_dry_raw.csv'))

    smd_thresholds = [0]
    sma_thresholds = [-5, -10, -12, -15, -17, -20]
    ndays = [5, 7, 10, 14]
    out_keys = []
    for smd_t, sma_t in itertools.product(smd_thresholds, sma_thresholds):
        k = 'd_smd{:03d}_sma{:02d}'.format(smd_t, sma_t)
        data.loc[:, k] = (data.loc[:, 'smd'] <= smd_t) & (data.loc[:, 'sma'] <= sma_t)
        out_keys.append(k)

    grouped_data = data.loc[:, ['month', 'year',
                                'smd', 'sma'] + out_keys].groupby(['month', 'year']).sum().reset_index()

    grouped_data.to_csv(os.path.join(event_def_dir, 'monthly_smd_dry_monthly_data.csv'))
    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
                                                                                        'monthly_smd_dry_monthly_data_desc.csv'))
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
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'monthly_smd_dry_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'monthly_smd_dry_prob_only_prob.csv'),
                                        float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'monthly_smd_dry_years.csv'))


def calc_dry_recurance_monthly_smd():
    data = get_vcsn_record(vcsn_version).reset_index()

    temp = calc_sma_smd_historical(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]
    data.to_csv(os.path.join(event_def_dir, 'dry_raw.csv'))

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

    grouped_data.to_csv(os.path.join(event_def_dir, 'dry_monthly_data.csv'))
    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
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
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'dry_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'dry_prob_only_prob.csv'), float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'dry_years.csv'))


def calc_wet_recurance():
    data = get_vcsn_record(vcsn_version).reset_index()
    temp = calc_sma_smd_historical(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]

    temp = False
    if temp:  # just to look at some plots
        fig, (ax, ax2, ax3) = plt.subplots(3, sharex=True)
        ax.plot(data.date, data.smd)
        ax2.plot(data.date, data.drain)
        ax3.plot(data.date, data.rain)
        plt.show()

    data.to_csv(os.path.join(event_def_dir, 'smd_wet_raw.csv'))

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

    grouped_data.to_csv(os.path.join(event_def_dir, 'smd_wet_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
                                                                                        'smd_wet_monthly_data_desc.csv'))

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
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'smd_wet_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'smd_wet_prob_only_prob.csv'),
                                        float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'smd_wet_years.csv'))


def calc_wet_recurance_ndays():
    ndays = {
        'org': {  # this is the best value!
            5: 14,
            6: 11,
            7: 11,
            8: 13,
            9: 13,
        }
    }
    for v in ndays.values():
        v.update({
            1: 99,
            2: 99,
            3: 99,
            4: 99,
            10: 99,
            11: 99,
            12: 99,
        })

    data = get_vcsn_record(vcsn_version).reset_index()
    temp = calc_sma_smd_historical(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]
    data.loc[:, 'ndays_rain'] = (data.loc[:, 'rain'] > 0.01).astype(float)
    data.to_csv(os.path.join(event_def_dir, 'ndays_wet_raw.csv'))

    grouped_data = data.loc[:, ['month', 'year', 'rain', 'ndays_rain']].groupby(['month', 'year']).sum().reset_index()

    grouped_data.to_csv(os.path.join(event_def_dir, 'ndays_wet_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
                                                                                        'ndays_wet_monthly_data_desc.csv'))

    # number of n days
    out_keys2 = []
    for k, val in ndays.items():
        ok = '{}'.format(k)
        out_keys2.append(ok)
        grouped_data.loc[:, 'limit'] = grouped_data.loc[:, 'month']
        grouped_data = grouped_data.replace({'limit': val})
        grouped_data.loc[:, ok] = grouped_data.loc[:, 'ndays_rain'] >= grouped_data.loc[:, 'limit']

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'ndays_wet_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'ndays_wet_prob_only_prob.csv'),
                                        float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'ndays_wet_years.csv'))


def calc_dry_rolling():
    bulk_ndays = [5, 10, 15, 20]
    ndays = {}
    for bnd in bulk_ndays:
        ndays['ndays{}'.format(bnd)] = {k: bnd for k in range(1, 13)}

    thresholds = {  # this did not end up getting used
        'first': {
            4: 15,
            5: 10,
            8: 5,
            9: 10,
        },
        'first-3': {
            4: 15 - 3,
            5: 10 - 3,
            8: 5 - 3,
            9: 10 - 3,
        },
        'first-5': {
            4: 15 - 5,
            5: 10 - 5,
            8: 5 - 5,
            9: 10 - 5,
        },
        'first-10': {
            4: 15 - 10,
            5: 10 - 10,
            8: 5 - 10,
            9: 10 - 10,
        },
        'zero': {
            4: 0,
            5: 0,
            8: 0,
            9: 0,
        },
        'one': {
            4: 1,
            5: 1,
            8: 1,
            9: 1,
        },
        'first-7': {
            4: 15 - 7,
            5: 10 - 7,
            8: 5 - 7,
            9: 10 - 7,
        },
    }
    for v in thresholds.values():
        v.update({
            1: -1,
            2: -1,
            3: -1,
            6: -1,
            7: -1,

            10: -1,
            11: -1,
            12: -1,
        })

    data = get_vcsn_record(vcsn_version).reset_index()
    data.loc[:, 'roll_rain_10'] = data.loc[:, 'rain'].rolling(10).sum()

    out_keys = []
    outdata = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 13), range(1972, 2020)], names=['month', 'year']))
    for nd, thresh in itertools.product(ndays.keys(), thresholds.keys()):
        temp_data = data.copy(deep=True)
        ok = '{}_{}'.format(thresh, nd)
        out_keys.append(ok)
        for m in range(1, 13):
            idx = data.month == m
            temp_data.loc[idx, ok] = temp_data.loc[idx, 'roll_rain_10'] <= thresholds[thresh][m]
            temp_data.loc[idx, 'ndays'] = ndays[nd][m]
        temp_data = temp_data.groupby(['month', 'year']).agg({ok: 'sum', 'ndays': 'mean'})
        outdata.loc[:, ok] = temp_data.loc[:, ok] >= temp_data.loc[:, 'ndays']

    outdata.to_csv(os.path.join(event_def_dir, 'rolling_dry_monthly.csv'))
    outdata = outdata.reset_index()

    out = outdata.loc[:, ['month'] + out_keys].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(outdata, set(out_keys) - set(drop_keys), out)
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'rolling_dry_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'variable_hot_prob_only_prob.csv'),
                                        float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'rolling_dry_years.csv'))
    return list(set(out_keys) - set(drop_keys)), out


def calc_dry_recurance_ndays():
    ndays = {  # happy with this value other than middle ones; this did not end up getting used
        'lower_q': {  # based on the sma -20 10days
            1: 31,  # lower quartile of normal
            2: 45,  # lower quartile of normal
            3: 38,  # lower quartile of normal

            4: 46,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 37,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 35,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 30,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month

            10: 53,  # lower quartile of normal
            11: 43,  # lower quartile of normal
            12: 47,  # lower quartile of normal
        },
        'up_5': {  # based on the sma -20 10days
            1: 31,  # lower quartile of normal
            2: 45,  # lower quartile of normal
            3: 38,  # lower quartile of normal

            4: 46 + 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 37 + 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 35 + 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 30 + 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month

            10: 53,  # lower quartile of normal
            11: 43,  # lower quartile of normal
            12: 47,  # lower quartile of normal
        },
        'down_5': {  # based on the sma -20 10days
            1: 31,  # lower quartile of normal
            2: 45,  # lower quartile of normal
            3: 38,  # lower quartile of normal

            4: 46 - 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 37 - 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 35 - 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 30 - 5,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month

            10: 53,  # lower quartile of normal
            11: 43,  # lower quartile of normal
            12: 47,  # lower quartile of normal
        },
        'down_7': {  # based on the sma -20 10days
            1: 31,  # lower quartile of normal
            2: 45,  # lower quartile of normal
            3: 38,  # lower quartile of normal

            4: 46 - 7,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 37 - 7,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 35 - 7,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 30 - 7,  # lower quartile of normal, pair with 'hot' as pet is imporant in this month

            10: 53,  # lower quartile of normal
            11: 43,  # lower quartile of normal
            12: 47,  # lower quartile of normal
        },
    }
    for v in ndays.values():
        v.update({
            6: -1,
            7: -1,
        })

    data = get_vcsn_record(vcsn_version).reset_index()
    temp = calc_sma_smd_historical(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]
    data.loc[:, 'ndays_rain'] = (data.loc[:, 'rain'] > 0.01).astype(float)
    data.to_csv(os.path.join(event_def_dir, 'ndays_dry_raw.csv'))

    grouped_data = data.loc[:, ['month', 'year', 'rain', 'ndays_rain']].groupby(['month', 'year']).sum().reset_index()

    grouped_data.to_csv(os.path.join(event_def_dir, 'ndays_dry_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
                                                                                        'ndays_dry_monthly_data_desc.csv'))

    # number of n days
    out_keys2 = []
    for k, val in ndays.items():
        ok = '{}'.format(k)
        out_keys2.append(ok)
        grouped_data.loc[:, 'limit'] = grouped_data.loc[:, 'month']
        grouped_data = grouped_data.replace({'limit': val})
        grouped_data.loc[:, ok] = grouped_data.loc[:, 'rain'] <= grouped_data.loc[:, 'limit']

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'ndays_dry_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'ndays_dry_prob_only_prob.csv'),
                                        float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'ndays_dry_years.csv'))


def calc_hot_recurance_variable():
    var_to_use = {
        1: 'tmax',
        2: 'tmax',
        3: 'tmax',
        4: 'tmean',  # to use in conjunction with dry to get atual dry
        5: 'tmean',  # to use in conjunction with dry to get atual dry
        6: 'tmax',
        7: 'tmax',
        8: 'tmean',  # to use in conjunction with dry to get atual dry
        9: 'tmean',  # to use in conjunction with dry to get atual dry
        10: 'tmax',
        11: 'tmax',
        12: 'tmax',

    }
    ndays = {
        '5day': {
            4: 5,
            5: 5,
            8: 5,
            9: 5,
        },
        '7day': {
            4: 7,
            5: 7,
            8: 7,
            9: 7,
        },
        '10day': {
            4: 10,
            5: 10,
            8: 10,
            9: 10,
        },
        '15day': {
            4: 15,
            5: 15,
            8: 15,
            9: 15,
        }
    }

    thresholds = {
        'upper_q': {  # based on the sma -20 10days

            4: 18,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 15,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 13,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 15,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month

        },
        '2_less': {  # based on the sma -20 10days

            4: 18 - 2,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 15 - 2,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 13 - 2,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 15 - 2,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month

        },
        '5_less': {  # based on the sma -20 10days

            4: 18 - 5,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 15 - 5,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 13 - 5,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 15 - 5,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month

        },
        '7_less': {  # based on the sma -20 10days

            4: 18 - 7,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            5: 15 - 7,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            8: 13 - 7,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month
            9: 15 - 7,  # upper quartile of normal, pair with 'hot' as pet is imporant in this month

        }
    }
    for v in thresholds.values():  # set for actual hot events
        v.update({
            1: 25,
            2: 25,
            3: 25,
            6: 25,
            7: 25,
            10: 25,
            11: 25,
            12: 25,
        })
    for v in ndays.values():  # set for actual hot events
        v.update({
            1: 7,
            2: 7,
            3: 7,
            6: 7,
            7: 7,
            10: 7,
            11: 7,
            12: 7,
        })

    data = get_vcsn_record(vcsn_version).reset_index()
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2

    out_keys = []
    outdata = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 13), range(1972, 2020)], names=['month', 'year']))
    for thresh, nd in itertools.product(thresholds.keys(), ndays.keys()):
        temp_data = data.copy(deep=True)
        ok = '{}_{}'.format(thresh, nd)
        out_keys.append(ok)
        for m in range(1, 13):
            idx = data.month == m
            temp_data.loc[idx, ok] = temp_data.loc[idx, var_to_use[m]] >= thresholds[thresh][m]
            temp_data.loc[idx, 'ndays'] = ndays[nd][m]
        temp_data = temp_data.groupby(['month', 'year']).agg({ok: 'sum', 'ndays': 'mean'})
        outdata.loc[:, ok] = temp_data.loc[:, ok] >= temp_data.loc[:, 'ndays']

    outdata.to_csv(os.path.join(event_def_dir, 'variable_hot_monthly.csv'))
    outdata = outdata.reset_index()

    out = outdata.loc[:, ['month'] + out_keys].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(outdata, set(out_keys) - set(drop_keys), out)
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'variable_hot_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'variable_hot_prob_only_prob.csv'),
                                        float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'variable_hot_years.csv'))


def joint_hot_dry():
    hot = pd.read_csv(os.path.join(event_def_dir, 'variable_hot_years.csv'), index_col=0)
    hot_keys = list(hot.keys())
    dry = pd.read_csv(os.path.join(event_def_dir, 'rolling_dry_years.csv'), index_col=0)
    dry_keys = list(dry.keys())
    data = pd.merge(hot, dry, left_index=True, right_index=True)
    use_data = []
    for d in data.keys():
        use_data.append(
            pd.Series([np.nan if isinstance(t, float) else tuple(int(e) for e in t.strip('()').split(',')) for t in
                       data.loc[:, d]]))

    use_data = pd.concat(use_data, axis=1)
    use_data.columns = data.columns
    _org_describe_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    _describe_names = []
    for e in _org_describe_names:
        _describe_names.extend(['{}_irr'.format(e), '{}_dry'.format(e)])
    full_event_names = ['hot:{}_dry:{}'.format(h, d) for h, d in itertools.product(hot_keys, dry_keys)]
    outdata = pd.DataFrame(index=pd.Series(range(1, 13), name='month'),
                           columns=pd.MultiIndex.from_product((full_event_names,
                                                               (['prob'] + _describe_names))
                                                              , names=['event', 'pga_desc']), dtype=float)
    # make base data
    print('making base data')
    for hot_nm, dry_nm in itertools.product(hot_keys, dry_keys):
        en = 'hot:{}_dry:{}'.format(hot_nm, dry_nm)
        joint_event = pd.Series(list(set(use_data.loc[:, hot_nm]).intersection(set(use_data.loc[:, dry_nm]))))
        if joint_event.dropna().empty:
            continue
        temp = make_prob(joint_event)
        outdata.loc[temp.index, (en, 'prob')] = temp.values[:, 0]
        temp = add_pga_from_idx(joint_event)
        outdata.loc[temp.index, (en, _describe_names)] = temp.loc[:, _describe_names].values
    t = pd.Series([' '.join(e) for e in outdata.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    outdata.loc[:, outdata.columns[idx]] *= 100
    outdata = outdata.sort_index(axis=1, level=0, sort_remaining=False)
    outdata.to_csv(os.path.join(event_def_dir, 'joint_hot_dry_prob.csv'), float_format='%.1f%%')

    idx = t.str.contains('prob')
    outdata.loc[:, outdata.columns[idx]].to_csv(os.path.join(event_def_dir, 'joint_hot_dry_prob_only_prob.csv'),
                                                float_format='%.1f%%')
    idx = t.str.contains('mean')
    outdata.loc[:, outdata.columns[idx]].to_csv(os.path.join(event_def_dir, 'joint_hot_dry_mean_impact.csv'),
                                                float_format='%.1f%%')
    return full_event_names, outdata


def make_prob(in_series):
    in_series = in_series.dropna()
    data = pd.DataFrame(np.atleast_2d(list(in_series.values)), columns=['month', 'year'])
    out_series = data.groupby('month').count() / 48
    return pd.DataFrame(out_series)


def old_calc_restrict_recurance():
    data = get_restriction_record()

    thresholds = [0.5, 0.75, 1]
    tnames = ['half', '3/4', 'full']
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

    grouped_data.to_csv(os.path.join(event_def_dir, 'rest_monthly_data.csv'))
    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
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
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    out_years.to_csv(os.path.join(event_def_dir, 'rest_years.csv'))

    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100
    out.to_csv(os.path.join(event_def_dir, 'old_rest_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'old_rest_prob_only_prob.csv'),
                                        float_format='%.1f%%')


def calc_restrict_cumulative_recurance():
    data = get_restriction_record()

    ndays = [1, 5, 7, 10, 14, 21, 25, 29]
    ndays = {'{:02d}'.format(e): e for e in ndays}
    temp = {1: 10,
            2: 17,
            3: 17,
            4: 10,
            5: 7,
            6: 10,
            7: 10,
            8: 10,
            9: 7,
            10: 5,
            11: 5,
            12: 7,
            }
    ndays['eqlikly'] = temp  # note don't use 'prob' in this name!

    grouped_data = data.loc[:, ['month', 'year', 'f_rest']].groupby(['month', 'year']).sum().reset_index()

    # make montly restriction anaomaloy - mean
    temp = grouped_data.groupby('month').mean().loc[:, 'f_rest'].to_dict()
    grouped_data.loc[:, 'f_rest_an_mean'] = grouped_data.loc[:, 'month']
    grouped_data = grouped_data.replace({'f_rest_an_mean': temp})
    grouped_data.loc[:, 'f_rest_an_mean'] = grouped_data.loc[:, 'f_rest'] - grouped_data.loc[:, 'f_rest_an_mean']

    # make montly restriction anaomaloy - median
    temp = grouped_data.groupby('month').median().loc[:, 'f_rest'].to_dict()
    grouped_data.loc[:, 'f_rest_an_med'] = grouped_data.loc[:, 'month']
    grouped_data = grouped_data.replace({'f_rest_an_med': temp})
    grouped_data.loc[:, 'f_rest_an_med'] = grouped_data.loc[:, 'f_rest'] - grouped_data.loc[:, 'f_rest_an_med']

    grouped_data.to_csv(os.path.join(event_def_dir, 'rest_monthly_data.csv'))
    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
                                                                                        'rest_monthly_data_desc.csv'))
    # number of n days
    out_keys2 = []
    for k, nd in ndays.items():
        ok = '{}d_rest'.format(k)
        out_keys2.append(ok)
        if isinstance(nd, int):
            grouped_data.loc[:, ok] = grouped_data.loc[:, 'f_rest'] >= nd
        elif isinstance(nd, dict):
            grouped_data.loc[:, ok] = grouped_data.loc[:, 'f_rest'] >= grouped_data.loc[:, 'month'].replace(nd)

        else:
            raise ValueError('unexpected type for nd: {}'.format(type(nd)))

    out = grouped_data.loc[:, ['month'] + out_keys2].groupby(['month']).aggregate(['sum', prob])
    drop_keys = []
    for k in out_keys2:
        temp = (out.loc[:, k].loc[:, 'sum'] == 48).all() or (
                out.loc[:, k].loc[:, 'sum'] == 0).all()
        if temp:
            drop_keys.append(k)

    out = out.drop(columns=drop_keys)
    out, out_years = add_pga(grouped_data, set(out_keys2) - set(drop_keys), out)
    out_years.to_csv(os.path.join(event_def_dir, 'rest_years.csv'))

    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100
    out.to_csv(os.path.join(event_def_dir, 'rest_prob.csv'), float_format='%.1f%%')
    idx = (t.str.contains('prob') | t.str.contains('sum'))
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'rest_prob_only_prob.csv'), float_format='%.1f%%')


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
        v.to_csv(os.path.join(event_def_dir, 'len_rest_{}_raw.csv'.format(k)))
        temp = v.groupby(['year', 'month']).agg({k: ['sum', 'count',
                                                     'mean', 'min', 'max']})
        temp = temp.rename(columns=rename_mapper, level=1)
        all_data = all_data.combine_first(temp)

    all_data = all_data.loc[:, (tnames, out_columns)]
    all_data.reset_index().astype(float).groupby('month').describe().to_csv(os.path.join(event_def_dir,
                                                                                         'len_rest_month_desc_no_zeros.csv'))
    t = all_data['any']['num_per'].isna().reset_index().groupby('month').agg({'num_per': ['sum', prob]})
    t.to_csv(os.path.join(event_def_dir, 'len_rest_prob_no_rest.csv'))
    all_data = all_data.fillna(0)
    all_data.to_csv(os.path.join(event_def_dir, 'len_rest_monthly.csv'))

    all_data.reset_index().groupby('month').describe().to_csv(
        os.path.join(event_def_dir, 'len_rest_month_desc_with_zeros.csv'))

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
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100
    out.to_csv(os.path.join(event_def_dir, 'len_rest_prob.csv'), float_format='%.1f%%')
    out_years.to_csv(os.path.join(event_def_dir, 'len_rest_years.csv'))
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'len_rest_prob_only_prob.csv'),
                                        float_format='%.1f%%')


def calc_cold_recurance():
    data = get_vcsn_record(vcsn_version)
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2
    data.loc[:, 'tmean_raw'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2
    data.loc[:, 'tmean'] = data.loc[:, 'tmean'].rolling(3).mean()
    data.to_csv(os.path.join(event_def_dir, 'rolling_cold_raw.csv'))

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

    grouped_data.to_csv(os.path.join(event_def_dir, 'rolling_cold_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
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
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'rolling_cold_prob.csv'), float_format='%.1f%%')
    out_years.to_csv(os.path.join(event_def_dir, 'rolling_cold_years.csv'))
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'rolling_cold_prob_only_prob.csv'),
                                        float_format='%.1f%%')


def calc_hot_recurance():
    data = get_vcsn_record(vcsn_version)
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2
    data.to_csv(os.path.join(event_def_dir, 'temp_raw.csv'))

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

    grouped_data.to_csv(os.path.join(event_def_dir, 'hot_monthly_data.csv'))

    grouped_data.drop(columns=['year']).groupby('month').describe().to_csv(os.path.join(event_def_dir,
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
    t = pd.Series([' '.join(e) for e in out.columns])
    idx = ~((t.str.contains('sum')) | (t.str.contains('count')))
    out.loc[:, out.columns[idx]] *= 100

    out.to_csv(os.path.join(event_def_dir, 'hot_prob.csv'), float_format='%.1f%%')
    out.loc[:, out.columns[idx]].to_csv(os.path.join(event_def_dir, 'hot_prob_only_prob.csv'), float_format='%.1f%%')

    out_years.to_csv(os.path.join(event_def_dir, 'hot_years.csv'))


def plot_vcsn_smd():
    data, use_cords = vcsn_pull_single_site(
        lat=-43.358,
        lon=172.301,
        year_min=1972,
        year_max=2019,
        use_vars=('evspsblpot', 'pr'))
    print(use_cords)

    temp = calc_sma_smd_historical(data['pr'], data['evspsblpot'], data.date, 150, 1)

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
    # final run set up
    calc_dry_recurance_monthly_smd()
    calc_dry_recurance()
    calc_hot_recurance()
    calc_cold_recurance()
    calc_wet_recurance_ndays()
    calc_restrict_cumulative_recurance()
