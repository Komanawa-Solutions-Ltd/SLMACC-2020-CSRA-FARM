"""
a quick test of using the moving block bootstrap for flows

 Author: Matt Hanson
 Created: 8/02/2021 12:36 PM
 """

from Climate_Shocks.Stochastic_Weather_Generator.moving_block_bootstrap import MovingBlockBootstrapGenerator
from Climate_Shocks.get_past_record import get_restriction_record
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
import project_base
import itertools
import pandas as pd
import numpy as np
from Climate_Shocks.climate_shocks_env import event_def_path

baseoutdir = os.path.join(project_base.unbacked_dir, 'gen_flow_test')

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
        temp = temp.loc[(temp.day <= month_len[m])].flow.values
        input_data[key] = temp
        sim_len[key] = month_len[m]
        assert (len(temp) % month_len[m]) == 0, 'problem with m{}'.format(m)
    nmonths_comments = ''
    for k, v in nmonths.items():
        nmonths_comments += '{}: {}\n '.format(k, v)
    return input_data, block, sim_len, nmonths_comments


def examine_auto_correlation(extension='.png'):
    outdir = os.path.join(baseoutdir, 'generator_plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_auto_correlation(10000, 15, k, show=False, hlines=[0, 0.25, 0.5, 0.75])
        fig.savefig(os.path.join(outdir, 'correlation_{}{}'.format(k, extension)))
        plt.close()


def examine_means(extension='.png'):
    outdir = os.path.join(baseoutdir, 'generator_plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    boot = get_irrigation_generator()

    for k in boot.keys:
        fig, ax = boot.plot_1d(k, include_input=True, bins=50, show=False)
        fig.savefig(os.path.join(outdir, 'mean_{}{}'.format(k, extension)))
        plt.close()
    for k in boot.keys:
        fig, ax = boot.plot_1d(k, include_input=False, bins=50, show=False, density=False)
        fig.savefig(os.path.join(outdir, 'not_density_mean_{}{}'.format(k, extension)))
        plt.close()


def get_irrigation_generator(recalc=False):
    nsims = 1e5
    nsims = int(nsims)
    input_data, block, sim_len, nmonths_comments = make_input_data_1month()
    comments = '''generator created by get irrigation generator {} to provide daily 
    irrigation timeseries data using the detrended historical irrigation data\n 
    number of months for each key:\n''' + nmonths_comments
    generator_path = os.path.join(baseoutdir, 'irrigation_gen.nc')
    boot = MovingBlockBootstrapGenerator(input_data=input_data, blocktype='truncnormal', block=block,
                                         nsims=nsims, data_path=generator_path, sim_len=sim_len, nblocksize=50,
                                         save_to_nc=True, comments=comments, recalc=recalc)
    return boot


def export_example_generator_data():

    gen = get_irrigation_generator()
    means = [50, 100, 150]
    key = 'm01-D'
    cmap = get_cmap('tab20')
    n_scens = 20
    colors = [cmap(e / n_scens) for e in range(n_scens)]  # pick from color map

    for m in means:
        data = gen.get_data(n_scens, key, m, 5)
        fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(14, 10))
        for i in range(n_scens):
            if i < n_scens/2:
                ax = ax1
            else:
                ax = ax2
            ax.plot(range(data.shape[1]), data[i], ls='-', alpha=1, c=colors[i])
            ax.set_xlabel('day')
            ax.set_ylabel('flow')
        fig.suptitle(f'{n_scens} example record for {key}')
        fig.tight_layout()
        fig.savefig(os.path.join(baseoutdir, 'generator_plots', f'example_data_{key}_mean_{m}.png'))


if __name__ == '__main__':
    # get_irrigation_generator()
    # examine_means()
    # examine_auto_correlation()
    export_example_generator_data()
