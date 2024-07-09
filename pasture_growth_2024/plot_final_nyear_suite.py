"""
created matt_dumont 
on: 7/8/24
"""
import project_base
import os
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from Pasture_Growth_Modelling.basgra_parameter_sets import default_mode_sites
from komanawa.slmacc_csra import get_nyr_non_exceedence_prob

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


def prob_non_exceed(nyr, figsize=(10, 8), suffix='.png'):
    outdir = os.path.join(base_outdir, 'prob_non_exceed')
    os.makedirs(outdir, exist_ok=True)
    data = {}
    max_pg = 0
    min_pg = np.inf
    for mode, site in default_mode_sites:
        t = pd.DataFrame(get_nyr_non_exceedence_prob(nyr, site, mode, None)).reset_index()
        t['pg'] *= 1 / 1000
        data[site, mode] = t
        max_pg = max(max_pg, t.pg.max())
        if mode != 'dryland':
            min_pg = min(min_pg, t.pg.min())

    ndecimals = np.ceil(np.log10(max_pg))
    max_pg = np.ceil(max_pg / 10 ** (ndecimals - 3)) * 10 ** (ndecimals - 3)
    ndecimals = np.ceil(np.log10(min_pg))
    min_pg = np.floor(min_pg / 10 ** (ndecimals - 3)) * 10 ** (ndecimals - 3)
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 0.1]))
    for i, site in enumerate(['oxford', 'eyrewell', 'oxford']):
        ax = axs[i, 0]
        legax = axs[i, 1]
        legax.axis('off')

        plt_mode = deepcopy(plt_modes)
        colors = deepcopy(all_colors)
        use_max = max_pg
        use_min = min_pg
        if site == 'eyrewell':
            plt_mode.remove('dryland')
            colors.remove('wheat')
        elif site == 'oxford' and i == 2:
            plt_mode.remove('dryland')
            colors.remove('wheat')
        else:
            plt_mode = ['dryland']
            colors = ['wheat']
            use_max = data[site, 'dryland'].pg.max()
            ndecimals = np.ceil(np.log10(use_max))
            use_max = np.ceil(use_max / 10 ** (ndecimals - 3)) * 10 ** (ndecimals - 3)
            use_min = data[site, 'dryland'].pg.min()
            ndecimals = np.ceil(np.log10(use_min))
            use_min = np.floor(use_min / 10 ** (ndecimals - 3)) * 10 ** (ndecimals - 3)

        # exceedence_prob
        for c, mode in zip(colors, plt_mode):
            exceed = data[site, mode]
            exceed = exceed.sort_values('pg')
            x = np.concatenate(([use_min], exceed.pg, [use_max]))
            y = np.concatenate(([0], exceed.prob, [100]))

            ax.plot(x, y, c=c, label=f'{site} {mode}'.title())
            ax.fill_between(x, 0, y, color=c, alpha=0.5)
        ax.set_xlim(use_min, use_max)
        legax.legend(*ax.get_legend_handles_labels(), loc='center left')

    fig.suptitle(f'Non-exceedance probability {nyr}-year sim')
    fig.supxlabel(f'Pasture growth (Tons DM / ha / {nyr} years)')
    fig.supylabel('Cumulative probability (%)')
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(outdir, f'prob_non_exceed_nyr{nyr:02d}{suffix}'))


if __name__ == '__main__':
    for nyr in [2, 3, 5, 10]:
        prob_non_exceed(nyr)
