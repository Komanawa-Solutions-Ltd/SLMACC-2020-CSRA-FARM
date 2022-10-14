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


# getting the target dataframes
basepath = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/stats_info/V3")
measured_full_df = pd.read_csv(basepath.joinpath("measured_full_stats.csv"))
measured_baseline_df = pd.read_csv(basepath.joinpath("measured_baseline_stats.csv"))
measured_climate_df = pd.read_csv(basepath.joinpath("measured_climate_stats.csv"))
nat_full_df = pd.read_csv(basepath.joinpath("naturalised_full_stats.csv"))
nat_baseline_df = pd.read_csv(basepath.joinpath("naturalised_baseline_stats.csv"))
nat_climate_df = pd.read_csv(basepath.joinpath("naturalised_climate_stats.csv"))

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
                      'short_filamentous_wua'
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



from scipy.stats import gaussian_kde

colours1 = ['midnightblue', 'lightskyblue']
colours2 = ['darkgreen', 'lightgreen']
colours3 = ['hotpink', 'pink']


def plot_barcharts(datasets, dataset_names):
    """Plotting bar charts from data frames onto an appropriate figure layout"""

    # initalise figures
    nrows, ncol = (2, 2)
    figsize = (12, 10)
    num_figs = len(variable_names_bar) // (nrows * ncol) + 1
    all_axes = []
    all_figs = []
    for n in range(num_figs):
        f, axs = plt.subplots(nrows, ncol, figsize=figsize)
        all_figs.append(f)
        all_axes.extend(axs.flatten())

    # plot data
    water_years = datasets[0]['water_year']
    for var, ax in zip(variable_names_bar, all_axes):
        x = (water_years - min(water_years)) * len(datasets)  # keynote this assumes datasets have same x
        for i, (df, n, c) in enumerate(zip(datasets, dataset_names, colours3)):
            ax.bar(x + i, df[var], color=c, label=n)
        for xi in x:
            ax.axvline(xi - 0.5, color='k', ls=':', alpha=0.5)
        ax.set_title(var.upper())
        ax.set_xticks(x + (len(datasets) - 1) / 2)
        ax.set_xticklabels(water_years, rotation=-60)
        ax.legend()

    # post plotting functions
    for i, f in enumerate(all_figs):
        outpath = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/stats_info/V3",
                                                        f'barplots_climate_{i:02d}_of_{len(all_figs):02d}.png')
        f.tight_layout()
        f.savefig(outpath)
    plt.show()


def plot_histograms(datasets, dataset_names, density=True):
    """plotting histograms """

    nrows, ncol = (2, 2)
    figsize = (12, 10)
    num_figs = len(variable_names_hist) // (nrows * ncol) + 1
    all_axes = []
    all_figs = []
    for n in range(num_figs):
        f, axs = plt.subplots(nrows, ncol, figsize=figsize)
        all_figs.append(f)
        all_axes.extend(axs.flatten())

    var_lims = {}
    for var in variable_names_hist:
        all_data = []
        for df in datasets:
            all_data.extend(df[var])
        var_lims[var] = (np.nanmin(all_data), np.nanmax(all_data))

    for var, ax in zip(variable_names_hist, all_axes):
        for i, (df, n, c) in enumerate(zip(datasets, dataset_names, colours3)):
            if density:
                try:
                    dens = gaussian_kde(df[var])
                    xs = np.linspace(*var_lims[var], 100)
                    ys = dens(xs)
                    ax.plot(xs, ys, c=c, label=n)
                    ax.fill_between(xs, ys, color=c, alpha=0.3)
                except np.linalg.LinAlgError:
                    pass
            else:
                ax.hist(df[var], color=c, bins=8, alpha=0.2)
        ax.set_title(var.upper())
        ax.legend()
    for i, f in enumerate(all_figs):
        outpath = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/stats_info/V3",
                                                        f'histplots_climate_{i:02d}_of_{len(all_figs):02d}.png')
        f.tight_layout()
        f.savefig(outpath)
    plt.show()




if __name__ == '__main__':
    datasets = [measured_climate_df, nat_climate_df]
    dataset_names = ['Measured_climate', 'Naturalised_climate']
    #colors = get_colors(datasets)
    plot_barcharts(datasets, dataset_names)
    plot_histograms(datasets, dataset_names)
    pass
