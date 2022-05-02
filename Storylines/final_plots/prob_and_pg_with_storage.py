import pandas as pd
from copy import deepcopy
import os
import ksl_env
from Storylines.storage_runs.run_hurt_scare_mostprob import change_to_daily_pg, default_mode_sites
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from Storylines.storyline_runs.run_random_suite import get_1yr_data
from Storylines.storyline_evaluation.storyline_characteristics_for_impact import get_exceedence, month_len

base_outdir = os.path.join(ksl_env.slmmac_dir, 'final_plots')
os.makedirs(base_outdir, exist_ok=True)

plt_modes = [
    'dryland',
    'irrigated',
    'store400',
    'store600',
    'store800',
]

all_colors = [
    'wheat',
    'lightcoral',
    'deepskyblue',
    'cornflowerblue',
    'darkblue',

]


def prob_non_exceed(site, figsize=(10, 8), suffix='.png'):
    plt_mode = deepcopy(plt_modes)
    colors = deepcopy(all_colors)
    if site == 'eyrewell':
        plt_mode.remove('dryland')
        colors.remove('wheat')
    outdir = os.path.join(base_outdir, 'prob_non_exceed')
    os.makedirs(outdir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)

    # exceedence_prob
    for c, mode in zip(colors, plt_mode):
        exceed = get_exceedence(site, mode, True)
        ax1.plot(np.concatenate((exceed.pg, [18])), np.concatenate((1 - exceed.prob, [1])), c=c,
                 label=f'{site} {mode}'.title())
        ax1.fill_between(np.concatenate((exceed.pg, [18])), 0, np.concatenate((1 - exceed.prob, [1])), color=c,
                         alpha=0.5)
    ax1.set_title('Non-exceedance probability')
    ax1.set_ylabel('Cumulative probability')
    ax1.legend()

    # histogram...
    for c, mode in zip(colors, plt_mode):
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
    fig.savefig(os.path.join(outdir, f'{site}_prob_non_exceed{suffix}'))


def pg_boxplots(site, figsize=(10, 8), suffix='.png'):
    plt_mode = deepcopy(plt_modes)
    colors = deepcopy(all_colors)
    if site == 'eyrewell':
        plt_mode.remove('dryland')
        colors.remove('wheat')
    outdir = os.path.join(base_outdir, 'pg_boxplots')
    os.makedirs(outdir, exist_ok=True)
    data = get_1yr_data(True, True, True).dropna()
    plot_keys = [f'pg_m{m:02d}' for m in [9, 10, 11, 12, 1, 2, 3, 4, 5]]
    plot_labs = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

    n_scens = len(plt_mode)
    initialpositions = np.arange(0, len(plot_keys) * n_scens, n_scens)
    fig, ax = plt.subplots(nrows=1, figsize=figsize)
    labels = []
    handles = []
    for i, (c, mode) in enumerate(zip(colors, plt_mode)):
        print(site, mode)
        labels.append(f'{site}-{mode}')
        handles.append(Patch(facecolor=c))

        # arrange data
        use_positions = initialpositions + i + 0.5
        if mode == 'dryland':
            mod = 5
        else:
            mod = 8
        if 'store' in mode:
            prob = 10 ** (data.loc[:, f'log10_prob_irrigated'] + mod)
        else:
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
    fig.savefig(os.path.join(outdir, f'{site}_pg_boxplot{suffix}'))  # svg too big


if __name__ == '__main__':
    sites = ['eyrewell', 'oxford']
    for site in sites:
        print(site)
        print('running prob_non_exceed')
        prob_non_exceed(site)
        print('running pg_boxplot')
        pg_boxplots(site)

