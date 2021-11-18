"""
 Author: Matt Hanson
 Created: 17/11/2021 12:31 PM
 """

import pandas as pd
import os
import ksl_env
from Storylines.storage_runs.run_hurt_scare_mostprob import change_to_daily_pg, default_mode_sites
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from Storylines.storyline_runs.run_random_suite import get_1yr_data
from Storylines.storyline_evaluation.storyline_characteristics_for_impact import get_exceedence, month_len

base_outdir = os.path.join(ksl_env.slmmac_dir, 'hydrosoc_plots')
os.makedirs(base_outdir, exist_ok=True)


def plot_normalize_storyline(name, figsize=(10, 8), suffix='.png'):
    print(name)
    outputs_dir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', name)

    datas = [pd.read_csv(os.path.join(outputs_dir, 'corrected_data.csv'), index_col=0),
             ]
    data_labels = ['corrected']
    for ij, (data, data_lab) in enumerate(zip(datas, data_labels)):
        print(data_lab)
        if ij < 2:
            data = change_to_daily_pg(data)
        plot_outdir = os.path.join(base_outdir, f'{data_lab}_storage_plots')
        os.makedirs(plot_outdir, exist_ok=True)
        data_describe = data.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95])

        plot_keys = [f'pg_m{m:02d}' for m in [9, 10, 11, 12, 1, 2, 3, 4, 5]] + ['pg_yr1']
        plot_labs = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Year\nMean']
        n_scens = (len(default_mode_sites) - 1) / 2
        initialpositions = np.arange(0, 10 * n_scens, n_scens)

        colors = {
            'irrigated': 'b',
            'store400': 'm',
            'store600': 'y',
            'store800': 'c',
        }

        # initialize plots
        all_plots = {}
        all_plot_names = ['Pasture Growth']
        for p in all_plot_names:
            axs = {
                'eyrewell': plt.subplots(figsize=figsize),
                'oxford': plt.subplots(figsize=figsize),
            }
            all_plots[p] = axs

        # nested box plots per site
        for site_over, (fig, ax) in all_plots['Pasture Growth'].items():
            i = 0
            for mode, site in default_mode_sites:
                if (site != site_over) or (mode == 'dryland'):
                    continue
                use_positions = initialpositions + i + 0.5
                i += 1
                c = colors[mode]
                plot_data = [data.loc[:, f'{site}-{mode}_{e}'].values for e in plot_keys]
                bp = ax.boxplot(plot_data, positions=use_positions, patch_artist=True)
                for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
                    plt.setp(bp[element], color='k')
                plt.setp(bp['fliers'], markeredgecolor=c)
                for patch in bp['boxes']:
                    patch.set(facecolor=c)

        # make pretty and save
        for plt_nm, plot_dict in all_plots.items():
            for site_over, (fig, ax) in plot_dict.items():
                ax.set_title(f'{plt_nm} {site_over.capitalize()}')
                if ij < 2:
                    ax.set_ylabel('Pasture Growth (kg dm/ha/day)')
                    ax.set_ylim(0, 100)
                else:
                    ax.set_ylabel('ratio to irrigate')
                    ax.set_ylim(0, 15)
                if plt_nm in ['Mean Pasture Growth',
                              'Mean Pasture Growth with Percentiles',
                              'Median Pasture Growth',
                              'Median Pasture Growth with Percentiles',
                              ]:
                    ax.set_xticks(range(0, 13))
                    ax.set_xticklabels(plot_labs)
                    ax.legend()
                else:
                    ax.set_xticks(initialpositions + n_scens // 2)
                    ax.set_xticklabels(plot_labs)
                    # set verticle lines
                    for i in np.concatenate((initialpositions, initialpositions + n_scens)):
                        ax.axvline(x=i,
                                   ymin=0,
                                   ymax=1,
                                   linestyle=':',
                                   color='k',
                                   alpha=0.5
                                   )

                    labels = []
                    handles = []
                    for mode, site in default_mode_sites:
                        if (site != site_over) or (mode == 'dryland'):
                            continue
                        if mode =='irrigated':
                            labels.append('irrigated (no storage)')
                        else:
                            labels.append(mode)
                        handles.append(Patch(facecolor=colors[mode]))
                    ax.legend(handles=handles, labels=labels)
                fig.tight_layout()
                fig.savefig(os.path.join(plot_outdir, f'{plt_nm}-{name}-{site_over}{suffix}'))


def prob_non_exceed(figsize=(10, 8), suffix='.png'):
    outdir = os.path.join(base_outdir, 'prob_non_exceed')
    os.makedirs(outdir, exist_ok=True)
    plt_mode_sites = [('dryland', 'oxford'),
                      ('irrigated', 'oxford'),
                      ('irrigated', 'eyrewell'),
                      ]
    colors = ['wheat', 'lightcoral', 'cornflowerblue']

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)

    # exceedence_prob
    for c, (mode, site) in zip(colors, plt_mode_sites):
        exceed = get_exceedence(site, mode, True)
        ax1.plot(np.concatenate((exceed.pg, [18])), np.concatenate((1 - exceed.prob, [1])), c=c,
                 label=f'{site} {mode}'.title())
        ax1.fill_between(np.concatenate((exceed.pg, [18])), 0, np.concatenate((1 - exceed.prob, [1])), color=c,
                         alpha=0.5)
    ax1.set_title('Non-exceedance probability')
    ax1.set_ylabel('Cumulative probability')
    ax1.legend()

    # histogram...
    for c, (mode, site) in zip(colors, plt_mode_sites):
        exceed = get_exceedence(site, mode, True)
        exceed.loc[:, 'up'] = (1 - exceed.prob).diff()
        exceed.dropna(inplace=True)
        x = np.concatenate(([0], exceed.pg, [18]))
        y = np.concatenate(([0], exceed.up, [0]))
        ax2.plot(x, y, c=c,
                 label=f'{site} {mode}'.title())
        ax2.fill_between(x, 0, y, color=c,
                         alpha=0.5)
    ax2.set_title('Probability density function')
    ax2.set_ylabel('Event probability')
    ax2.set_xlabel('Pasture growth tons/ha/yr')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'prob_non_exceed{suffix}'))


def pg_boxplots(figsize=(10, 8), suffix='.png'):
    outdir = os.path.join(base_outdir, 'pg_boxplots')
    os.makedirs(outdir, exist_ok=True)
    data = get_1yr_data(True, True, True).dropna()
    plot_keys = [f'pg_m{m:02d}' for m in [9, 10, 11, 12, 1, 2, 3, 4, 5]]
    plot_labs = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

    plt_mode_sites = [('dryland', 'oxford'),
                      ('irrigated', 'oxford'),
                      ('irrigated', 'eyrewell'),
                      ]
    colors = ['wheat', 'lightcoral', 'cornflowerblue']
    n_scens = len(plt_mode_sites)
    initialpositions = np.arange(0, len(plot_keys) * n_scens, n_scens)
    fig, ax = plt.subplots(nrows=1, figsize=figsize)
    labels = []
    handles = []
    for i, (c, (mode, site)) in enumerate(zip(colors, plt_mode_sites)):
        print(site, mode)
        labels.append(f'{site}-{mode}')
        handles.append(Patch(facecolor=c))

        # arrange data
        use_positions = initialpositions + i + 0.5
        if mode == 'dryland':
            mod = 5
        else:
            mod = 8
        prob = 10 ** (data.loc[:, f'log10_prob_{mode}'] + mod)
        prob = prob / prob.sum()
        np.random.seed(5575)
        idx = np.random.choice(data.index, int(1e6), p=prob)
        plot_data = [data.loc[idx, f'{site}-{mode}_{e}'].values / month_len[int(e[-2:])] for e in plot_keys]
        bp = ax.boxplot(plot_data, positions=use_positions, patch_artist=True)
        for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='k')
        plt.setp(bp['fliers'], markeredgecolor=c)
        for patch in bp['boxes']:
            patch.set(facecolor=c)

    ax.set_xticks(initialpositions + n_scens // 2)
    ax.set_xticklabels(plot_labs)
    # set verticle lines
    for i in np.concatenate((initialpositions, initialpositions + n_scens)):
        ax.axvline(x=i,
                   ymin=0,
                   ymax=1,
                   linestyle=':',
                   color='k',
                   alpha=0.5
                   )

    ax.legend(handles=handles, labels=labels)

    ax.set_ylabel('Pasture Growth (kg dm/ha/day)')
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'pg_boxplot.png')) # svg too big


if __name__ == '__main__':
    fs = (6.5, 6)
    sf = '.svg'
    #prob_non_exceed(figsize=(10.7, 5.7), suffix=sf)
    #pg_boxplots(figsize=(9.3, 7.3), suffix=sf)
    if True:
        for n in ['storage_hurt', 'storage_scare', 'storage_most_probable']:
            plot_normalize_storyline(name=n, figsize=fs, suffix=sf)
