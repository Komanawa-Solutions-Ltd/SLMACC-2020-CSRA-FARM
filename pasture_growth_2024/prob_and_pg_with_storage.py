import pandas as pd
from copy import deepcopy
import os
import project_base
from Storylines.storage_runs.run_hurt_scare_mostprob import change_to_daily_pg, default_mode_sites
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from Storylines.storyline_evaluation.storyline_characteristics_for_impact import month_len
from komanawa.slmacc_csra import get_1yr_non_exceedence_prob, get_1yr_data

base_outdir = os.path.join(project_base.slmmac_dir, '0_Y2_and_Final_Reporting', 'final_plots')
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


def prob_non_exceed(figsize=(10, 8), suffix='.png'):
    outdir = os.path.join(base_outdir, 'prob_non_exceed')
    os.makedirs(outdir, exist_ok=True)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 0.1]))
    for i, site in enumerate(['eyrewell', 'oxford']):
        ax = axs[i, 0]
        legax = axs[i, 1]
        legax.axis('off')

        plt_mode = deepcopy(plt_modes)
        colors = deepcopy(all_colors)
        if site == 'eyrewell':
            plt_mode.remove('dryland')
            colors.remove('wheat')

        # exceedence_prob
        for c, mode in zip(colors, plt_mode):
            exceed = pd.DataFrame(get_1yr_non_exceedence_prob(site, mode, None)).reset_index()
            exceed = exceed.sort_values('pg')
            x = np.concatenate(([0], exceed.pg, [20000]))
            y = np.concatenate(([0], exceed.prob, [100]))

            ax.plot(x, y, c=c, label=f'{site} {mode}'.title())
            ax.fill_between(x, 0, y, color=c, alpha=0.5)
        ax.set_title(f'{site.capitalize()} Non-exceedance probability')
        ax.set_xlim(0, 19000)
        legax.legend(*ax.get_legend_handles_labels(), loc='center left')

    fig.supxlabel('Pasture growth (kg dm/ha/year)')
    fig.supylabel('Cumulative probability')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'prob_non_exceed{suffix}'))


def pg_boxplots(site, figsize=(10, 8), suffix='.png'):
    plt_mode = deepcopy(plt_modes)
    colors = deepcopy(all_colors)
    if site == 'eyrewell':
        plt_mode.remove('dryland')
        colors.remove('wheat')
    outdir = os.path.join(base_outdir, 'pg_boxplots')
    os.makedirs(outdir, exist_ok=True)
    plot_keys = [f'pg_m{m:02d}' for m in [9, 10, 11, 12, 1, 2, 3, 4, 5]]
    plot_labs = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

    n_scens = len(plt_mode)
    initialpositions = np.arange(0, len(plot_keys) * n_scens, n_scens)
    fig, ax = plt.subplots(nrows=1, figsize=figsize)
    labels = []
    handles = []
    for i, (c, mode) in enumerate(zip(colors, plt_mode)):
        print(site, mode)
        data = get_1yr_data(site=site, mode=mode).dropna()
        labels.append(f'{site}-{mode}')
        handles.append(Patch(facecolor=c))

        # arrange data
        use_positions = initialpositions + i + 0.5

        plot_data = [data.loc[:, f'{e}'].values / month_len[int(e[-2:])] for e in plot_keys]
        bp = ax.boxplot(plot_data, positions=use_positions, patch_artist=True,
                        whis=(0, 100), showfliers=False)
        for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='k')
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
    prob_non_exceed()
    for site in sites:
        print(site)
        pg_boxplots(site)

        plt.close('all')
