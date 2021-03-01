"""
 Author: Matt Hanson
 Created: 8/02/2021 12:36 PM
 """

from Climate_Shocks.Stochastic_Weather_Generator.moving_block_bootstrap import MovingBlockBootstrapGenerator
from Climate_Shocks.get_past_record import get_restriction_record
import matplotlib.pyplot as plt
import os
import ksl_env
import itertools
import pandas as pd
import numpy as np
from Climate_Shocks.climate_shocks_env import event_def_path

baseoutdir = os.path.join(ksl_env.slmmac_dir_unbacked, 'gen_vfinal')

month_len = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def make_input_data_2months():
    # decided not enough data here

    input_data = {}
    sim_len = {}
    nmonths = {}
    org_data = get_restriction_record('detrended', recalc=False).set_index(['year', 'month'])
    event_data = pd.read_csv(event_def_path, skiprows=1)
    event_data.loc[:, 'month2'] = event_data.loc[:, 'month']
    event_data = event_data.set_index(['year', 'month'])

    for precip, m in itertools.product(['ND-ND', 'ND-D', 'D-ND', 'D-D'],
                                       month_len.keys()):
        key = 'm{:02d}-{}'.format(m, precip)
        if precip.split('-')[0] == 'ND':
            firstp = [0, -1]
        else:
            firstp = [1]
        if precip.split('-')[1] == 'ND':
            sndp = [0, -1]
        else:
            sndp = [1]

        yearmonths = event_data.loc[np.in1d(event_data.precip, sndp) &
                                    np.in1d(event_data.prev_precip, firstp) &
                                    (event_data.month2 == m)]
        n = len(yearmonths)
        if n == 0:
            continue
        nmonths[key] = n
        yearmonths = yearmonths.index.values

        temp = org_data.loc[yearmonths]
        temp = temp.loc[(temp.day <= month_len[m])].f_rest.values
        input_data[key] = temp
        sim_len[key] = month_len[m]
        assert (len(temp) % month_len[m]) == 0, 'problem with m{}'.format(m)
    block = (7, 1, 5, 11)
    nmonths_comments = ''
    for k, v in nmonths.items():
        nmonths_comments += '{}: {}\n '.format(k, v)
    return input_data, block, sim_len, nmonths_comments


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
    org_data = get_restriction_record('detrended', recalc=False).set_index(['year', 'month'])
    event_data = pd.read_csv(event_def_path, skiprows=1)
    event_data.loc[:, 'month2'] = event_data.loc[:, 'month']
    event_data = event_data.set_index(['year', 'month'])

    for precip, m in itertools.product(['ND', 'D'], month_len.keys()):
        key = 'm{:02d}-{}'.format(m, precip)
        if precip == 'ND':
            firstp = [0, -1]
        else:
            firstp = [1]

        yearmonths = event_data.loc[np.in1d(event_data.precip, firstp) &
                                    (event_data.month2 == m)]
        n = len(yearmonths)
        if n == 0:
            continue
        nmonths[key] = n
        yearmonths = yearmonths.index.values

        temp = org_data.loc[yearmonths]
        temp = temp.loc[(temp.day <= month_len[m])].f_rest.values
        input_data[key] = temp
        sim_len[key] = month_len[m]
        assert (len(temp) % month_len[m]) == 0, 'problem with m{}'.format(m)
    nmonths_comments = ''
    for k, v in nmonths.items():
        nmonths_comments += '{}: {}\n '.format(k, v)
    return input_data, block, sim_len, nmonths_comments


def examine_auto_correlation():
    outdir = os.path.join(baseoutdir, 'generator_plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_auto_correlation(10000, 15, k, show=False, hlines=[0, 0.25, 0.5, 0.75])
        fig.savefig(os.path.join(outdir, 'correlation_{}.png'.format(k)))
        plt.close()


def examine_means():
    outdir = os.path.join(baseoutdir, 'generator_plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_means(k, include_input=True, bins=50, show=False)
        fig.savefig(os.path.join(outdir, 'mean_{}.png'.format(k)))
        plt.close()
    for k in boot.keys:
        fig, ax = boot.plot_means(k, include_input=False, bins=50, show=False, density=False)
        fig.savefig(os.path.join(outdir, 'not_density_mean_{}.png'.format(k)))
        plt.close()



def get_irrigation_generator():
    nsims = 1e7
    nsims = int(nsims)
    input_data, block, sim_len, nmonths_comments = make_input_data_1month()
    comments = '''generator created by get irrigation generator {} to provide daily 
    irrigation timeseries data using the detrended historical irrigation data\n 
    number of months for each key:\n''' + nmonths_comments
    generator_path = os.path.join(baseoutdir, 'irrigation_gen.nc')
    boot = MovingBlockBootstrapGenerator(input_data=input_data, blocktype='truncnormal', block=block,
                                         nsims=nsims, data_path=generator_path, sim_len=sim_len, nblocksize=50,
                                         save_to_nc=True, comments=comments)
    return boot


if __name__ == '__main__':

    # gen_v1: 2 month precip with # block = (5, 1.5, 2, 8) v1 plots are range(nlags so x axis is shifted one value to the left e.g. x=0 should be x=1
    # gen_v2: 2 month precip with # block = (7, 1, 5, 11)
    # gen_v3: 1 month precip with all block = (7, 1, 5, 11)
    # gen_v4: 1 month precip with multiple blocks see nc file
    # gen_v5: 1 month precip with multiple blocks see nc file
    # gen_v6: 1 month precip with multiple blocks see nc file

    get_irrigation_generator()
