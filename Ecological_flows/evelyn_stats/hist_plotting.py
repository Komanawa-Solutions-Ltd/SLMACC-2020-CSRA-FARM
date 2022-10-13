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

# getting the target dataframes
basepath = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/stats_info/V3")
measured_full_df = pd.read_csv(basepath.joinpath("final_stats_measured_full.csv"))
measured_baseline_df = pd.read_csv(basepath.joinpath("final_stats_measured_baseline.csv"))
measured_climate_df = pd.read_csv(basepath.joinpath("final_stats_measured_climate.csv"))
nat_full_df = pd.read_csv(basepath.joinpath("final_stats_nat_full.csv"))
nat_baseline_df = pd.read_csv(basepath.joinpath("final_stats_nat_baseline.csv"))
nat_climate_df = pd.read_csv(basepath.joinpath("final_stats_nat_climate.csv"))



# listing the variable names of the columns to plot

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

variable_names_hist = ['longfin_eel_score', 'shortfin_eel_score',
                       'torrent_fish_score', 'common_bully_score',
                       'upland_bully_score',
                       'bluegill_bully_score',
                       'food_production_score',
                       'brown_trout_adult_score',
                       'chinook_salmon_junior_score',
                       'diatoms_score',
                       'long_filamentous_score',
                       'short_filamentous_score',
                       'black_fronted_tern_score',
                       'wrybill_plover_score',
                       'days_below_malf_score',
                       'days_below_flow_lim_score',
                       'anomalies_score',
                       'malf_events_greater_7_score',
                       'malf_events_greater_14_score',
                       'malf_events_greater_21_score',
                       'malf_events_greater_28_score',
                       'flow_events_greater_7_score',
                       'flow_events_greater_14_score',
                       'flow_events_greater_21_score',
                       'flow_events_greater_28_score',
                       'temp_days_above_19_score',
                       'temp_days_above_21_score',
                       'temp_days_above_24_score']



def plot_barcharts(datasets, dataset_names, colors):
    """Plotting bar charts from data frames onto an appropriate figure layout"""

    nrows, ncol = (2, 2)
    figsize = (12, 10)
    num_figs = len(variable_names_bar) // (nrows * ncol) + 1
    all_axes = []
    all_figs = []
    for n in range(num_figs):
        f, axs = plt.subplots(nrows, ncol, figsize=figsize)
        all_figs.append(f)
        all_axes.extend(axs.flatten())

    water_years = datasets[0]['water_year']
    for var, ax in zip(variable_names_bar, all_axes):
        outpath = kslcore.KslEnv.shared_gdrive.joinpath(f"Z2003_SLMACC/eco_modelling/stats_info/V3/.f{var}")
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

def plot_histograms(datasets, dataset_names, colors):
    """plotting histograms """


# todo EC work through this

# example
    from scipy.stats import gaussian_kde

    data = [1.5] * 7 + [2.5] * 2 + [3.5] * 8 + [4.5] * 3 + [5.5] * 1 + [6.5] * 8
    density = gaussian_kde(data)
    xs = np.linspace(0, 8, 200)
    ys = density(xs)


    nrows, ncol = (2, 2)
    figsize = (12, 10)
    num_figs = len(variable_names_hist) // (nrows * ncol) + 1
    all_axes = []
    all_figs = []
    for n in range(num_figs):
        f, axs = plt.subplots(nrows, ncol, figsize=figsize)
        all_figs.append(f)
        all_axes.extend(axs.flatten())

    water_years = datasets[0]['water_year']
    for var, ax in zip(variable_names_hist, all_axes):
        x = (water_years - min(water_years)) * len(datasets)
        for i, (df, n, c) in enumerate(zip(datasets, dataset_names, colors)):
            density = gaussian_kde(df[var])


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




if __name__ == '__main__':
    datasets = [measured_climate_df, nat_climate_df]  # todo this is a holder for all of your 6 datasets, make in sets of years
    dataset_names = ['Measured_climate', 'Naturalised_climate']  # todo this is a holder for all of your 6 datasets
    colors = get_colors(datasets)
    plot_barcharts(datasets, dataset_names, colors)
    #pass

