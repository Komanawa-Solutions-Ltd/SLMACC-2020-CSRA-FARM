"""
 Author: Matt Hanson
 Created: 8/02/2021 12:36 PM
 """

from Climate_Shocks.Stochastic_Weather_Generator.moving_block_bootstrap import MovingBlockBootstrapGenerator
from Climate_Shocks.get_past_record import get_restriction_record
import matplotlib.pyplot as plt
import os
import project_base
import itertools
import pandas as pd
import numpy as np
from Climate_Shocks.climate_shocks_env import event_def_path
from copy import deepcopy
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_smd_monthly
from Climate_Shocks import climate_shocks_env
from BS_work.SWG.SWG_wrapper import get_monthly_smd_mean_detrended

baseoutdir = project_base.get_irrigation_gen_vfinal()

month_len = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def get_trended_classified():
    data = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 13), range(1972, 2020)], names=['month', 'year']),
                        columns=['temp_class', 'precip_class', 'rest', 'rest_cum'], dtype=float)

    vcsn = get_vcsn_record('trended')
    vcsn.loc[:, 'sma'] = calc_smd_monthly(vcsn.rain, vcsn.pet, vcsn.index) - vcsn.loc[:, 'doy'].replace(
        get_monthly_smd_mean_detrended(leap=False, recalc=True))
    vcsn.loc[:, 'tmean'] = (vcsn.loc[:, 'tmax'] + vcsn.loc[:, 'tmin']) / 2
    vcsn.loc[:, 'month_mapper'] = vcsn.loc[:, 'month'].astype(int)

    # make, save the cutoffs for use in checking functions!
    vcsn = vcsn.groupby(['month', 'year']).mean()
    upper_limit = pd.read_csv(os.path.join(climate_shocks_env.supporting_data_dir, 'upper_limit.csv'), index_col=0)
    lower_limit = pd.read_csv(os.path.join(climate_shocks_env.supporting_data_dir, 'lower_limit.csv'), index_col=0)

    rest_rec = get_restriction_record('trended').groupby(['month', 'year']).sum().loc[:, 'f_rest']

    data.loc[:, ['rest', 'rest_cum']] = 0
    data.loc[:, ['temp_class', 'precip_class']] = 'A'
    data.loc[rest_rec.index, 'rest_cum'] = rest_rec
    data = pd.merge(data, vcsn, right_index=True, left_index=True)
    data.loc[:, 'rest_cum'] = [rc / month_len[m] for rc, m in
                               data.loc[:, ['rest_cum', 'month_mapper']].itertuples(False, None)]

    # set hot
    var = 'tmean'
    idx = data.loc[:, var] >= data.loc[:, 'month_mapper'].replace(upper_limit.loc[:, var].to_dict())
    data.loc[idx, 'temp_class'] = 'H'

    # set cold
    var = 'tmean'
    idx = data.loc[:, var] <= data.loc[:, 'month_mapper'].replace(lower_limit.loc[:, var].to_dict())
    data.loc[idx, 'temp_class'] = 'C'

    # set wet
    var = 'sma'  # negative is dry positive is wet
    idx = data.loc[:, var] >= data.loc[:, 'month_mapper'].replace(upper_limit.loc[:, var].to_dict())
    data.loc[idx, 'precip_class'] = 'W'

    # set dry
    var = 'sma'  # negative is dry positive is wet
    idx = data.loc[:, var] <= data.loc[:, 'month_mapper'].replace(lower_limit.loc[:, var].to_dict())
    data.loc[idx, 'precip_class'] = 'D'

    # re-order data
    data = data.reset_index().sort_values(['year', 'month'])

    return data


def make_input_data_1month():
    block = {
        'm01-D': (7, 1, 5, 11),
        'm01-ND': (9, 1.5, 7, 15),

        'm02-D': (8, 1, 5, 11),
        'm02-ND': (11, 1, 8, 15),

        'm03-D': (7, 1, 5, 11),
        'm03-ND': (8, 1, 5, 12),

        'm04-D': (8, 1, 5, 12),
        'm04-ND': (7, 1, 5, 11),

        'm05-D': (10, 1, 7, 15),
        'm05-ND': (8, 1, 6, 11),

        'm09-D': (9, 2, 5, 12),
        'm09-ND': (10, 2, 7, 15),

        'm10-D': (7, 1, 5, 11),
        'm10-ND': (10, 1, 8, 15),

        'm11-D': (11, 1, 7, 15),
        'm11-ND': (8, 1, 5, 11),

        'm12-D': (8, 1, 5, 12),
        'm12-ND': (8, 1, 5, 11),
    }
    input_data = {}
    sim_len = {}
    nmonths = {}
    org_data = get_restriction_record('trended', recalc=False).set_index(['year', 'month'])
    event_data = get_trended_classified()
    event_data.loc[:, 'month2'] = event_data.loc[:, 'month']
    event_data = event_data.set_index(['year', 'month'])

    for precip, m in itertools.product(['ND', 'D'], month_len.keys()):
        if m in [6, 7, 8]:
            continue
        key = 'm{:02d}-{}'.format(m, precip)
        if precip == 'ND':
            firstp = ['W', 'A']
        else:
            firstp = ['D']

        yearmonths = event_data.loc[np.in1d(event_data.precip_class, firstp) &
                                    (event_data.month2 == m)]
        n = len(yearmonths)
        if n == 0:
            continue
        nmonths[key] = n
        yearmonths = yearmonths.index.values

        temp = org_data.loc[yearmonths]
        temp = temp.loc[(temp.day <= month_len[m])].flow.values
        input_data[key] = temp
        sim_len[key] = month_len[m]
        assert (len(temp) % month_len[m]) == 0, 'problem with m{}'.format(m)
    nmonths_comments = ''
    for k, v in nmonths.items():
        nmonths_comments += '{}: {}\n '.format(k, v)
    return input_data, block, sim_len, nmonths_comments


def examine_auto_correlation(ext='png'):
    outdir = os.path.join(baseoutdir, f'generator_plots_{ext}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_auto_correlation(10000, 15, k, show=False, hlines=[0, 0.25, 0.5, 0.75])
        fig.savefig(os.path.join(outdir, 'correlation_{}.{}'.format(k, ext)))
        plt.close()


def examine_means(ext='png'):
    outdir = os.path.join(baseoutdir, f'generator_plots_{ext}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_1d(k, include_input=True, bins=50, show=False)
        fig.savefig(os.path.join(outdir, 'mean_{}.{}'.format(k, ext)))
        plt.close()
    for k in boot.keys:
        fig, ax = boot.plot_1d(k, include_input=False, bins=50, show=False, density=False)
        fig.savefig(os.path.join(outdir, 'not_density_mean_{}.{}'.format(k, ext)))
        plt.close()


def get_irrigation_generator(recalc=False):
    if recalc:
        raise InterruptedError('should not need to recalc, should pull from large_working')
    nsims = 1e7
    nsims = int(nsims)
    input_data, block, sim_len, nmonths_comments = make_input_data_1month()
    comments = '''generator created by get irrigation generator {} to provide daily 
    river flow timeseries data using the historical river flow data\n 
    monthly average irrigation restrictions under current allocations are produced to aid selection of river flow data\n
    number of months for each key:\n''' + nmonths_comments
    generator_path = os.path.join(baseoutdir, 'irrigation_gen.nc')
    boot = MovingBlockBootstrapGenerator(input_data=input_data, blocktype='truncnormal', block=block,
                                         nsims=nsims, data_path=generator_path, sim_len=sim_len, nblocksize=50,
                                         save_to_nc=True, comments=comments, recalc=recalc, seed=654685)
    boot.create_1d(make_restriction_mean, suffix='rest', pass_if_exists=True)

    return boot


def make_current_restrictions(data):
    # see simpligying assumptions
    # https://docs.google.com/document/d/1fCsGuHHEgFTcPl289Eboczmjqf5_4eLGTot6Zgxowr0/edit?usp=sharing
    out_data = deepcopy(data)
    out_data[data >= 63] = 0
    out_data[data <= 41] = 1
    idx = (data < 63) & (data > 41)
    out_data[idx] = (data[idx] * 501.86 - 20576) / 1000 / 11.041

    return out_data


def make_restriction_mean(data):
    out_data = make_current_restrictions(data)
    return out_data.mean(axis=1)


def check_rests():
    rests = get_restriction_record()
    rests.loc[:, 'calc_rest'] = make_current_restrictions(rests.loc[:, 'flow'].values)
    rests.loc[:, 'dif'] = rests['f_rest'] - rests['calc_rest']
    print(rests.dif.describe())


if __name__ == '__main__':
    # gen_v1: 2 month precip with # block = (5, 1.5, 2, 8) v1 plots are range(nlags so x axis is shifted one value to the left e.g. x=0 should be x=1
    # gen_v2: 2 month precip with # block = (7, 1, 5, 11)
    # gen_v3: 1 month precip with all block = (7, 1, 5, 11)
    # gen_v4: 1 month precip with multiple blocks see nc file
    # gen_v5: 1 month precip with multiple blocks see nc file
    # gen_v6: 1 month precip with multiple blocks see nc file
    for e in ['png', 'svg']:
        examine_means(e)
        examine_auto_correlation(e)
