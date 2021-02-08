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


def make_input_data():  # todo DD, DA breaks?, still need to decide, I would prefer to do this, but minimally d/nd
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

    input_data = {}
    sim_len = {}
    nmonths = {}
    org_data = get_restriction_record('detrended', recalc=False).set_index(['year', 'month'])
    event_data = pd.read_csv(event_def_path, skiprows=1)
    event_data.loc[:, 'month2'] = event_data.loc[:, 'month']
    event_data = event_data.set_index(['year', 'month'])

    for precip, m in itertools.product(['ND-ND', 'ND-D', 'D-ND', 'D-D'], month_len.keys()):
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

        # todo check how many years to sample from, and export and save.
        temp = org_data.loc[yearmonths]
        temp = temp.loc[(temp.day <= month_len[m])].f_rest.values
        input_data[key] = temp
        sim_len[key] = month_len[m]
        assert (len(temp) % month_len[m]) == 0, 'problem with m{}'.format(m)
    # block = (5, 1.5, 2, 8) #todo not sold on this exactly, review in  v1 remember x is off by one in auto correlation plots
    block = (7, 1, 5, 11)
    return input_data, block, sim_len


def examine_auto_correlation():
    outdir = os.path.join(ksl_env.slmmac_dir_unbacked, 'generator_plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_auto_correlation(10000, 15, k, show=False, hlines=[0, 0.25, 0.5, 0.75])
        fig.savefig(os.path.join(outdir, 'correlation_{}.png'.format(k)))
        plt.close()


def examine_means():
    outdir = os.path.join(ksl_env.slmmac_dir_unbacked, 'generator_plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_means(k, include_input=True, bins=50, show=False)
        fig.savefig(os.path.join(outdir, 'mean_{}.png'.format(k)))
        plt.close()


def get_irrigation_generator():
    nsims = 1e5  # todo inital to look at breaking up by d-d, nd-d
    nsims = int(nsims)
    input_data, block, sim_len = make_input_data()
    comments = '''generator created by get irrigation generator {} to provide daily 
    irrigation timeseries data using the detrended historical irrigation data'''
    generator_path = os.path.join(ksl_env.slmmac_dir_unbacked, 'irrigation_gen.nc')
    boot = MovingBlockBootstrapGenerator(input_data=input_data, blocktype='truncnormal', block=block,
                                         nsims=nsims, data_path=generator_path, sim_len=sim_len, nblocksize=50,
                                         save_to_nc=True, comments=comments)
    return boot


if __name__ == '__main__':
    # todo start by reviewing plots
    examine_auto_correlation()
    examine_means()
