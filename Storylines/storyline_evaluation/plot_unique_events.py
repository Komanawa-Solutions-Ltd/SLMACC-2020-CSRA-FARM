"""
created matt_dumont 
on: 23/06/22
"""
import itertools

import ksl_env
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from Ecological_flows.v2.random_suite import get_colors
from Pasture_Growth_Modelling.basgra_parameter_sets import default_mode_sites

outdir_base = Path(ksl_env.slmmac_dir).joinpath('0_Y2_and_Final_Reporting/final_plots/unique_events')
outdir_base.mkdir(exist_ok=True)
data_dir = Path(ksl_env.slmmac_dir).joinpath('outputs_for_ws', 'norm', 'unique_events_v2')
figsize = (16.5, 9.25)

month_to_month = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec',

}


def plot_unique_events(single_plots=False):
    if single_plots:
        outdir = outdir_base.joinpath('single_plots')
    else:
        outdir = outdir_base.joinpath('overview_plots')
    outdir.mkdir(exist_ok=True)
    for mode, site in default_mode_sites:
        sm = f'{site}-{mode}'
        data = pd.read_csv(data_dir.joinpath(f'{sm}-PGR-daily_total-singe_events.csv'), index_col=0)

        # normalise data to base year
        cols = np.arange(0, 24).astype(str)
        for m in data.month.unique():
            idx = data.month == m
            print(m)
            if m in [5, 6, 7, 8]:
                base_key = f'm{m:02d}-A-A-0-{sm}'
            else:
                base_key = f'm{m:02d}-A-A-50-{sm}'

            data.loc[idx, cols] += -1 * data.loc[base_key, cols]
        use_months = [9, 10, 11, 12, 1, 2, 3, 4, 5, ]
        if not single_plots:
            fig, axs = plt.subplots(3, 3, figsize=figsize)
        else:
            axs = np.array(use_months)
        ks = [f'{p}-{t}' for p, t in itertools.product(['W', 'A', 'D'], ['C', 'A', 'H'])]
        for m, ax in zip(use_months, axs.flatten()):
            if single_plots:
                fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(month_to_month[m])
            idx = data.month == m
            colors = {'H': 'r', 'C': 'b', 'A': 'grey'}
            lss = {'D': ':', 'W': '--', 'A': '-'}
            markers = {5: "v", 25: "v", 50: 's', 0: 's', 75: "^", 95: "^"}
            small = 2
            big = 4
            marker_sizes = {5: small, 25: big, 50: big, 0: big, 75: small, 95: big}

            use_cols = np.arange(12).astype(str)
            for i in data.index[idx]:

                temp = data.loc[i, 'temp']
                precip = data.loc[i, 'precip']
                rest = data.loc[i, 'rest']
                if mode == 'dryland':
                    if rest != 0 and rest != 50:
                        continue
                c = colors[temp]
                ls = lss[precip]
                marker = markers[rest]
                marker_size = marker_sizes[rest]
                ax.plot(use_cols.astype(int), data.loc[i, use_cols], c=c, ls=ls, ms=marker_size * 2,
                        marker=marker, alpha=0.5)
                if single_plots:
                    legend = True
                elif m == 5:
                    legend = True
                else:
                    legend = False

                if legend:
                    legend_elements = []
                    legend_elements.append(Patch(color='w', label='Precip class'))
                    for k, v in lss.items():
                        legend_elements.append(Line2D([0], [0], ls=v, label=k, color='k'))

                    legend_elements.append(Patch(color='w', label='Temp class'))
                    for k, v in colors.items():
                        legend_elements.append(Patch(color=v, label=k))

                    legend_elements.append(Patch(color='w', label='Restriction'))
                    for k, v in markers.items():
                        ms = marker_sizes[k]
                        legend_elements.append(Line2D([0], [0], marker=v, markersize=ms, label=k, color='k'))
                    ax.legend(handles=legend_elements, ncol=2, loc=4)
            if single_plots:
                fig.suptitle(sm.capitalize())
                fig.savefig(outdir.joinpath(f'{sm}-{month_to_month[m]}_unique.png'))
        if not single_plots:
            fig.suptitle(sm.capitalize())
            fig.tight_layout()
            fig.savefig(outdir.joinpath(f'{sm}_unique.png'))

    # todo plot winter months (no variation) for oxford and eyrewell


if __name__ == '__main__':
    plot_unique_events(single_plots=True)
    plot_unique_events(single_plots=False)
