"""
created Evelyn_Charlesworth 
on: 12/10/2022
"""

# plotting histograms of the variables of outdata for various sets of data

import kslcore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_colors
import datetime
from dateutil.relativedelta import relativedelta
from itertools import groupby
from kslcore import KslEnv
from Climate_Shocks.get_past_record import get_vcsn_record
# from water_temp_monthly import temp_regr
import ecological_scoring

# help how can I import the df directly from ecological_scoring rather than reading it in?

# getting the target dataframe
pathway = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/stats_info/final_stats_rounded.csv")
df = pd.read_csv(pathway)

# testing plotting hists


variable_names_bar = ['days_below_malf',
                      'temp_days_above_19', 'temp_days_above_21', 'temp_days_above_24',
                      'malf_consec_days', 'malf_num_events',
                      'malf_events_greater_7', 'malf_events_greater_14',
                      'malf_events_greater_21', 'malf_events_greater_28', 'days_below_50',
                      'flow_limits_consec_days', 'flow_limits_num_events', 'flow_events_greater_7',
                      'flow_events_greater_14',
                      'flow_events_greater_21', 'flow_events_greater_28', 'anomaly_1',
                      'longfin_eel_wua', 'shortfin_eel_wua', 'torrent_fish_wua',
                      'common_bully_wua', 'upland_bully_wua', 'bluegill_bully_wua',
                      'food_production_wua', 'brown_trout_adult_wua',
                      'chinook_salmon_junior_wua', 'diatoms_wua', 'long_filamentous_wua',
                      'short_filamentous_wua', 'black_fronted_tern_wua', 'wrybill_plover_wua',
                      ]

variable_names_hist = ['longfin_eel_score_rounded', 'shortfin_eel_score_rounded',
                       'torrent_fish_score_rounded', 'common_bully_score_rounded',
                       'upland_bully_score_rounded',
                       'bluegill_bully_score_rounded',
                       'food_production_score_rounded',
                       'brown_trout_adult_score_rounded',
                       'chinook_salmon_junior_score_rounded',
                       'diatoms_score_rounded',
                       'long_filamentous_score_rounded',
                       'short_filamentous_score_rounded',
                       'black_fronted_tern_score_rounded',
                       'wrybill_plover_score_rounded',
                       'days_below_malf_score_rounded',
                       'days_below_flow_lim_score_rounded',
                       'anomalies_score_rounded',
                       'malf_events_greater_7_score_rounded',
                       'malf_events_greater_14_score_rounded',
                       'malf_events_greater_21_score_rounded',
                       'malf_events_greater_28_score_rounded',
                       'flow_events_greater_7_score_rounded',
                       'flow_events_greater_14_score_rounded',
                       'flow_events_greater_21_score_rounded',
                       'flow_events_greater_28_score_rounded',
                       'temp_days_above_19_score_rounded',
                       'temp_days_above_21_score_rounded',
                       'temp_days_above_24_score_rounded']


def plot_barcharts(datasets, dataset_names, colors):

    nrows, ncol = (3, 2)
    figsize = (12, 9)
    num_figs = len(variable_names_bar) // (nrows * ncol) + 1
    all_axes = []
    all_figs = []
    for n in range(num_figs):
        f, axs = plt.subplots(nrows, ncol, figsize=figsize)
        all_figs.append(f)
        all_axes.extend(axs.flatten())

    water_years = datasets[0]['water_year']
    for var, ax in zip(variable_names_bar, all_axes):
        x = (water_years - min(water_years)) * len(datasets)  # keynote this assumes datasets have same x
        for i, (df, n, c) in enumerate(zip(datasets, dataset_names, colors)):
            ax.bar(x + i, df[var], color=c, label=n)
        for xi in x:
            ax.axvline(xi - 0.5, color='k', ls=':', alpha=0.5)
        ax.set_title(var.upper())
        ax.set_xticks(x + (len(datasets) - 1) / 2)
        ax.set_xticklabels(water_years, rotation=-60)
        ax.legend()

    for f in all_figs:
        f.tight_layout()

    plt.show()




# todo EC work thourgh this

# example
from scipy.stats import gaussian_kde

data = [1.5] * 7 + [2.5] * 2 + [3.5] * 8 + [4.5] * 3 + [5.5] * 1 + [6.5] * 8
density = gaussian_kde(data)
xs = np.linspace(0, 8, 200)
ys = density(xs)

for i, (df, n, c) in enumerate(zip(datasets, dataset_names, colors)):  # todo inside for loop for variable
    # todo calculate density here
    ax.plot()
    ax.fill_between()

fig1, axes1 = plt.subplots(7, 4, sharex=True, figsize=(10, 12))

for var1, ax1 in zip(variable_names_hist, axes1.ravel()):
    ax1.hist(df[var1], color='g', bins=8)
    ax1.set_title(var1.upper())
    ax1.set_xlim(-3.5, 3.5)
fig.tight_layout()
plt.show()

# for var in variable_names_bar:
#    plt.bar(df['water_year'], df['days_below_malf'])
#    fig, axes = fig, axes + 1


# for var in variable_names_bar:
#    sns.set_color_codes('deep')
#    plt.figure(figsize=(15, 20))
#    sns.barplot(data=df, x='water_year', y=var, color='g', ax=axes)
#    plt.show()
#
#
# for var1 in variable_names_hist:
#    sns.set_color_codes('deep')
#    plt.figure(figsize=(15, 20))
#    sns.histplot(data=df, x=var1, color='g', bins=8)
#    #plt.show()

# testing getting multiple on one grid

pass

if __name__ == '__main__':
    temp_df = df.copy(deep=True)
    temp_df.loc[:, variable_names_bar] *= 1 / 2
    datasets = [df, temp_df]  # todo this is a holder for all of your 6 datasets, make in sets of years
    dataset_names = ['original', 'half']  # todo this is a holder for all of your 6 datasets
    colors = get_colors(datasets)
    plot_barcharts(datasets, dataset_names, colors)