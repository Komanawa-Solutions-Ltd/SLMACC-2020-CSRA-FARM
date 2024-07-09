"""
 Author: Matt Hanson
 Created: 1/06/2021 11:22 AM
 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import warnings
warnings.warn('this code is not up to date with the current data, see komanawa-slmacc-csra for the most recent version')


def plot_1_yr_storylines(storylines, title, outdir=None, show=False, ax=None):
    """

    :param storylines: list of 1 year storylines (pd.DataFrames with an index other than month)
    :param outdir:
    :param show:
    :param plot_rest:
    :return:
    """

    temp_data, precip_data, rest_data = {}, {}, {}
    plt_months = np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
    for m in plt_months:
        temp_data[m] = []
        precip_data[m] = []
        rest_data[m] = []
    mapper = {'C': 0., 'A': 1., 'H': 2., 'W': 0., 'D': 2.}
    for m in plt_months:
        for story in storylines:
            story = story.set_index('month')
            temp_data[m].append(mapper[story.loc[m, 'temp_class']])
            precip_data[m].append(mapper[story.loc[m, 'precip_class']])
            if m in [5, 6, 7, 8]:
                rest_data[m] = 0
            else:
                rest_data[m].append(story.loc[m, 'rest'] * 3)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    t = ax.violinplot([temp_data[e] for e in plt_months], positions=np.arange(1, 13) - 0.33, widths=0.25,
                      showextrema=False)
    p = ax.violinplot([precip_data[e] for e in plt_months], positions=np.arange(1, 13), widths=0.25, showextrema=False)
    r = ax.violinplot([rest_data[e] for e in plt_months], positions=np.arange(1, 13) + 0.33, widths=0.25,
                      showextrema=False)

    ax.vlines(np.arange(2, 13) - 0.5, 0, 2, colors='k', linestyles=':')
    [patch.set_facecolor('lightcoral') for patch in t['bodies']]
    [patch.set_facecolor('cornflowerblue') for patch in p['bodies']]
    [patch.set_facecolor('lightseagreen') for patch in r['bodies']]
    [patch.set_alpha(1) for patch in t['bodies']]
    [patch.set_alpha(1) for patch in p['bodies']]
    [patch.set_alpha(1) for patch in r['bodies']]

    ax.set_ylim(0, 2)
    ax.set_xlim(0.5, 14)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['C, W, 0%', 'A, A, 50%', 'H, D, 100%'])
    ax.set_ylabel('Monthly conditions for temp_class, precip_class, rest')
    ax.set_xlabel('Month')
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels([str(e) for e in plt_months])
    ax.set_title(title.capitalize())
    ax.yaxis.grid(True)
    ax.legend(handles=[
        Patch(facecolor='lightcoral', label='temp_class'),
        Patch(facecolor='cornflowerblue', label='precip_class'),
        Patch(facecolor='lightseagreen', label='rest %'),
    ])

    fig.tight_layout()

    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig(os.path.join(outdir, f'{title}.png'))
    if show:
        plt.show()

    for m in plt_months:
        rest_data[m] = np.array(rest_data[m]) / 3
        temp_data[m] = np.array(temp_data[m])
        precip_data[m] = np.array(precip_data[m])
    return temp_data, precip_data, rest_data


if __name__ == '__main__':
    base_dir_trend = r"D:\mh_unbacked\SLMACC_2020_norm\temp_storyline_files\historical_quantified_1yr_trend"
    base_dir_detrend = r"D:\mh_unbacked\SLMACC_2020_norm\temp_storyline_files\historical_quantified_1yr_detrend"
    small_random = r"D:\mh_unbacked\SLMACC_2020_norm\temp_storyline_files\random_bad_irr"
    for base_dir, title in zip([base_dir_trend, base_dir_detrend, small_random], ['trend', 'detrend', 'small_rand']):
        paths = [os.path.join(base_dir, e) for e in os.listdir(base_dir)]
        storylines = [pd.read_csv(e) for e in paths]
        plot_1_yr_storylines(storylines, title, show=False)

    plt.show()
