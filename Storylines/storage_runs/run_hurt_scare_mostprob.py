"""
 Author: Matt Hanson
 Created: 15/10/2021 9:20 AM
 """
import shutil
import zipfile

import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import project_base
import os
import itertools
import glob
from copy import deepcopy
from Storylines.storyline_building_support import default_mode_sites
from Storylines.storyline_evaluation.storyline_eval_support import month_len
from Climate_Shocks import climate_shocks_env
from Pasture_Growth_Modelling.full_pgr_model_mp import default_pasture_growth_dir, run_full_model_mp, pgm_log_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth
from Pasture_Growth_Modelling.plot_full_model import plot_sims
from Pasture_Growth_Modelling.export_to_csvs import export_all_in_pattern
from Pasture_Growth_Modelling.full_model_implementation import out_variables
from Storylines.storyline_evaluation.transition_to_fraction import corr_pg
from Storylines.storyline_runs.run_random_suite import get_1yr_data
import time


def extract_zipfiles(src, dest, re_extract=False):
    if re_extract:
        if os.path.exists(dest):
            shutil.rmtree(dest)
    if not os.path.exists(dest):
        zipfile.ZipFile(src).extractall(dest)


def run_zipped_storylines(name, description_mult, zfile_path, number, choice_seed, pgr_seed, mode_sites, re_run,
                          run_pgr, run_export, run_mp,
                          start_run=0, nsims=100):
    """
    run from zipped storylines

    took 1.8602656841278076 min to run 16 parallel (on dickie, 16 threads) with 10 sims for 9 mode-sites
    took 5.193933200836182 min to run 16 parallel (on dickie, 16 threads) with 100 sims for 9 mode-sites
    took 40.957241813341774 min to run 16 parallel (on dickie, 16 threads) with 1000 sims for 9 mode-sites

    :param name: name for the run defines the location of outputs
    :param description_mult: description to be added to the netcdfs
    :param zfile_path: path to the zipfile containing the runs
    :param number: number of stories from teh zipfile to run, or 'all', which runs all
    :param choice_seed: seed for the choice of stories
    :param pgr_seed: seed for the pasture growth modelling
    :param mode_sites: mode and sites to be run for
    :param re_run: bool re-run passed to run pature growth modelle
    :param run_pgr: bool if True run the pasture growth model
    :param run_export: bool if true run the export process
    :param plot: bool if True run the plotting process
    :param run_mp: bool if True run as a multi process, else run individually (for debugging)
    :param start_run: the run to start on (int)
    :param nsims: number of sims for the pasture growth model per storyline
    :return:
    """
    base_pg_outdir = os.path.join(default_pasture_growth_dir, name)
    os.makedirs(base_pg_outdir, exist_ok=True)

    if run_pgr:
        temp_sls = os.path.join(climate_shocks_env.temp_storyline_dir, f'extracted_{name}')
        extract_zipfiles(zfile_path, temp_sls)
        path_list = np.array(glob.glob(os.path.join(temp_sls, '**/*.csv')))  # exclude the directory itself
        if isinstance(number, int):
            if number > len(path_list):
                raise ValueError(f'tried to get {number} scenarios from a set with only {len(path_list)} avalible')
            np.random.seed(choice_seed)
            path_list = np.random.choice(path_list, number, False)
        elif number == 'all':
            pass
        else:
            raise ValueError(f'weird value for number {number} should be "all" or int')

        outdirs = [base_pg_outdir for e in path_list]

        if run_mp:
            run_full_model_mp(
                storyline_path_mult=path_list,
                outdir_mult=outdirs,
                nsims_mult=nsims,
                log_path=os.path.join(pgm_log_dir, name),
                description_mult=description_mult,
                padock_rest_mult=False,
                save_daily_mult=False,
                verbose=False,
                re_run=re_run,
                mode_sites_mult=mode_sites,
                seed=pgr_seed,
                use_1_seed=False

            )
        else:
            for i, (p, od) in enumerate(zip(path_list, outdirs)):
                if i < start_run:
                    continue
                print(p)
                run_pasture_growth(storyline_path=p, outdir=od, nsims=nsims, padock_rest=False,
                                   save_daily=True, description='', verbose=True,
                                   n_parallel=1, re_run=True, seed=pgr_seed, use_1_seed=False,
                                   use_out_variables=out_variables, mode_sites=mode_sites
                                   )
    if run_export:
        outputs_dir = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'norm', name)
        os.makedirs(outputs_dir, exist_ok=True)

        export_all_in_pattern(base_outdir=os.path.join(outputs_dir, 'raw'),
                              patterns=[
                                  os.path.join(base_pg_outdir, '*.nc'),
                              ],
                              outvars=['m_' + e for e in out_variables], inc_storylines=False, agg_fun=np.nanmean,
                              mode_sites=mode_sites
                              )

        # the corrections
        columns = []
        plot_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        for mode, site in mode_sites:
            columns.extend([f'{site}-{mode}_pg_m{m:02d}' for m in plot_months])
            columns.append(f'{site}-{mode}_pg_yr1')

        temp_data = pd.read_csv(os.path.join(outputs_dir, 'raw', f'{site}-{mode}', 'm_PGR.csv'),
                                skiprows=1)  # just to get index
        index = sorted((temp_data.loc[:, 'month'].str.split('-').str[0] +
                        '-' +
                        temp_data.loc[:, 'month'].str.split('-').str[1]).values)

        outdata = pd.DataFrame(columns=columns)
        for mode, site in mode_sites:
            temp_data = pd.read_csv(os.path.join(outputs_dir, 'raw', f'{site}-{mode}', 'm_PGR.csv'),
                                    skiprows=1)  # just to get index
            temp_data.index = (temp_data.loc[:, 'month'].str.split('-').str[0] +
                               '-' +
                               temp_data.loc[:, 'month'].str.split('-').str[1]).values
            for m in plot_months:
                outdata.loc[:, f'{site}-{mode}_pg_m{m:02d}'] = temp_data.loc[:, str(m)] * month_len[m]
        outdata = corr_pg(outdata, mode_site=mode_sites)  # this calcs the 1 year
        outdata.to_csv(os.path.join(outputs_dir, 'corrected_data.csv'))
        outdata.mean().to_csv(os.path.join(outputs_dir, 'mean_corrected_data.csv'))


def plot_normalize_storyline(name, norm, plot):
    print(name)
    outputs_dir = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'norm', name)
    if norm:
        corrected_data = pd.read_csv(os.path.join(outputs_dir, 'corrected_data.csv'), index_col=0)
        corrected_data.index.name = 'ID'
        norm_data = pd.DataFrame(index=corrected_data.index, columns=corrected_data.columns)
        ratio_data = pd.DataFrame(index=corrected_data.index, columns=corrected_data.columns)
        old_data = get_old_data(corrected_data, name)
        for mode, site in default_mode_sites:
            if mode == 'dryland':
                continue

            base_key = f'{site}-irrigated'
            for m in range(1, 13):
                key = f'{site}-{mode}_pg_m{m:02d}'
                norm_data.loc[:, key] = corrected_data.loc[:, key] * (
                        old_data.loc[:, f'{base_key}_pg_m{m:02d}'] / corrected_data.loc[:, f'{base_key}_pg_m{m:02d}'])

                ratio_data.loc[:, key] = corrected_data.loc[:, key] / corrected_data.loc[:, f'{base_key}_pg_m{m:02d}']

            key = f'{site}-{mode}_pg_yr1'
            norm_data.loc[:, key] = corrected_data.loc[:, key] * (
                    old_data.loc[:, f'{base_key}_pg_yr1'] / corrected_data.loc[:, f'{base_key}_pg_yr1'])
            ratio_data.loc[:, key] = corrected_data.loc[:, key] / corrected_data.loc[:, f'{base_key}_pg_yr1']

        norm_data.to_csv(os.path.join(outputs_dir, 'normalised_corrected.csv'))
        ratio_data.to_csv(os.path.join(outputs_dir, 'ratio_corrected.csv'))

    if plot:
        datas = [pd.read_csv(os.path.join(outputs_dir, 'corrected_data.csv'), index_col=0),
                 pd.read_csv(os.path.join(outputs_dir, 'normalised_corrected.csv'), index_col=0),
                 pd.read_csv(os.path.join(outputs_dir, 'ratio_corrected.csv'), index_col=0),
                 ]
        data_labels = ['corrected', 'normalised_and_corrected', 'ratio to irrigated']
        for ij, (data, data_lab) in enumerate(zip(datas, data_labels)):
            print(data_lab)
            if ij < 2:
                data = change_to_daily_pg(data)
            plot_outdir = os.path.join(outputs_dir, f'{data_lab}_plots')
            os.makedirs(plot_outdir, exist_ok=True)
            data_describe = data.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95])

            plot_keys = [f'pg_m{m:02d}' for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]] + ['pg_yr1']
            plot_labs = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'year mean']
            n_scens = (len(default_mode_sites) - 1) / 2
            initialpositions = np.arange(0, 13 * n_scens, n_scens)

            colors = {
                'irrigated': 'b',
                'store400': 'm',
                'store600': 'y',
                'store800': 'c',
            }

            # initialize plots
            all_plots = {}
            all_plot_names = ['Mean Pasture Growth',
                              'Mean Pasture Growth with Percentiles',
                              'Median Pasture Growth',
                              'Median Pasture Growth with Percentiles',
                              'Pasture Growth Boxplot',
                              'Pasture Growth Violinplot'
                              ]
            for p in all_plot_names:
                axs = {
                    'eyrewell': plt.subplots(figsize=(10, 8)),
                    'oxford': plt.subplots(figsize=(10, 8)),
                }
                all_plots[p] = axs

            # overlayed mean plot for each site (pg)
            for site_over, (fig, ax) in all_plots['Mean Pasture Growth'].items():
                for mode, site in default_mode_sites:
                    if (site != site_over) or (mode == 'dryland'):
                        continue
                    x = range(len(plot_keys))
                    y = data_describe.loc['mean', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ax.plot(x, y, c=colors[mode], label=mode)

            # overlayed mean with percentiles plot for each site (pg)
            for site_over, (fig, ax) in all_plots['Mean Pasture Growth with Percentiles'].items():
                for mode, site in default_mode_sites:
                    if (site != site_over) or (mode == 'dryland'):
                        continue
                    x = range(len(plot_keys))
                    y = data_describe.loc['mean', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ax.plot(x, y, c=colors[mode], label=mode)
                    ytop = data_describe.loc['25%', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ybot = data_describe.loc['75%', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ax.fill_between(x, ybot, ytop, color=colors[mode], alpha=0.5, label=f'25th-75th {mode}')

            # overlayed median plot for each site (pg)
            for site_over, (fig, ax) in all_plots['Median Pasture Growth'].items():
                for mode, site in default_mode_sites:
                    if (site != site_over) or (mode == 'dryland'):
                        continue
                    x = range(len(plot_keys))
                    y = data_describe.loc['50%', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ax.plot(x, y, c=colors[mode], label=mode)

            # overlayed median with percentiles plot for each site (pg)
            for site_over, (fig, ax) in all_plots['Median Pasture Growth with Percentiles'].items():
                for mode, site in default_mode_sites:
                    if (site != site_over) or (mode == 'dryland'):
                        continue
                    x = range(len(plot_keys))
                    y = data_describe.loc['50%', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ax.plot(x, y, c=colors[mode], label=mode)
                    ytop = data_describe.loc['25%', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ybot = data_describe.loc['75%', [f'{site}-{mode}_{e}' for e in plot_keys]]
                    ax.fill_between(x, ybot, ytop, color=colors[mode], alpha=0.5, label=f'25th-75th {mode}')

            # individaul pg violin plots
            for site_over, (fig, ax) in all_plots['Pasture Growth Violinplot'].items():
                i = 0
                for mode, site in default_mode_sites:
                    if (site != site_over) or (mode == 'dryland'):
                        continue
                    use_positions = initialpositions + i + 0.5
                    i += 1
                    c = colors[mode]
                    plot_data = [data.loc[:, f'{site}-{mode}_{e}'].values for e in plot_keys]
                    parts = ax.violinplot(plot_data, positions=use_positions,
                                          showmeans=False, showmedians=True,
                                          quantiles=[[0.25, 0.75] for e in plot_data],
                                          )
                    for pc in parts['bodies']:
                        pc.set_facecolor(c)
                    parts['cmedians'].set_color(c)
                    parts['cquantiles'].set_color(c)
                    parts['cmins'].set_color(c)
                    parts['cmaxes'].set_color(c)
                    parts['cbars'].set_color(c)

            # nested box plots per site
            for site_over, (fig, ax) in all_plots['Pasture Growth Boxplot'].items():
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
                            labels.append(mode)
                            handles.append(Patch(facecolor=colors[mode]))
                        ax.legend(handles=handles, labels=labels)
                    fig.tight_layout()
                    fig.savefig(os.path.join(plot_outdir, f'{plt_nm}-{site_over}.png'))


def change_to_daily_pg(data):
    assert isinstance(data, pd.DataFrame)
    outdata = data.copy()
    for k in data.columns:
        if 'pg' not in k:
            continue
        suffix = k.split('_')[-1]
        if 'm' in suffix:
            mod = month_len[int(suffix.replace('m', ''))]
        elif 'yr' in suffix:
            mod = 365
        else:
            raise ValueError("shouldn't get here")
        outdata.loc[:, k] *= 1 / mod
    return outdata


def get_old_data(new_data, name):
    old_data = get_1yr_data(True, True, True)
    if name == 'storage_most_probable':
        old_data.loc[:, 'ID'] = old_data.loc[:, 'ID'] + '_' + old_data.loc[:, 'irr_type'] + '_irr'

    else:
        old_data.loc[:, 'ID'] = old_data.loc[:, 'ID'] + '_' + old_data.loc[:, 'irr_type']
    old_data.set_index('ID', inplace=True)
    old_data = old_data.loc[new_data.index]
    return old_data


def most_probabable(run_pgr, run_export):
    t = time.time()
    number = 8121  # 10% of c. 81,000
    nsims = 100
    name = 'storage_most_probable'
    description_mult = 'the most probable/baseline for the storage runs'
    zip_path = os.path.join(project_base.proj_root, 'Storylines/final_storylines/Final_most_probable.zip')

    run_zipped_storylines(
        name=name,
        description_mult=description_mult,
        zfile_path=zip_path,
        number=number,
        choice_seed=5548,
        pgr_seed=348357,
        mode_sites=default_mode_sites,
        re_run=False,
        run_pgr=run_pgr,
        run_export=run_export,
        run_mp=True,
        start_run=0, nsims=nsims)

    print(f'{name} took {(time.time() - t) / 60} min to run {number} parallel with 10 sims for '
          f'{len(default_mode_sites)} mode-sites')


def scare(run_pgr, run_export):
    t = time.time()
    number = 'all'  # 103 stories
    nsims = 100
    name = 'storage_scare'
    description_mult = 'scare scenarios with all storage systems'
    zip_path = os.path.join(project_base.proj_root,
                            'Storylines/final_storylines/Final_scare_autumn_drought2_mon_thresh_all.zip')

    run_zipped_storylines(
        name=name,
        description_mult=description_mult,
        zfile_path=zip_path,
        number=number,
        choice_seed=5548,
        pgr_seed=348357,
        mode_sites=default_mode_sites,
        re_run=True,
        run_pgr=run_pgr,
        run_export=run_export,
        run_mp=True,
        start_run=0, nsims=nsims)

    print(f'{name} took {(time.time() - t) / 60} min to run {number} parallel with 10 sims for '
          f'{len(default_mode_sites)} mode-sites')


def hurt(run_pgr, run_export):
    t = time.time()
    number = 'all'  # 226 stories
    nsims = 100
    name = 'storage_hurt'
    description_mult = 'hurt scenarios with all storage systems'
    zip_path = os.path.join(project_base.proj_root,
                            'Storylines/final_storylines/Final_hurt_hurt_v1_storylines_cluster_004.zip')

    run_zipped_storylines(
        name=name,
        description_mult=description_mult,
        zfile_path=zip_path,
        number=number,
        choice_seed=5548,
        pgr_seed=348357,
        mode_sites=default_mode_sites,
        re_run=True,
        run_pgr=run_pgr,
        run_export=run_export,
        run_mp=True,
        start_run=0, nsims=nsims)

    print(f'{name} took {(time.time() - t) / 60} min to run {number} parallel with 10 sims for '
          f'{len(default_mode_sites)} mode-sites')


if __name__ == '__main__':
    pgr = False
    export = False
    norm = True
    plot = True

    scare(pgr, export)
    hurt(pgr, export)
    most_probabable(pgr, export)

    for n in ['storage_hurt', 'storage_scare', 'storage_most_probable']:
        plot_normalize_storyline(name=n, norm=norm, plot=plot)
