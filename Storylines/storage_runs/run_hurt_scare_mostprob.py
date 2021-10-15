"""
 Author: Matt Hanson
 Created: 15/10/2021 9:20 AM
 """
import shutil
import zipfile
import pandas as pd
import numpy as np
import ksl_env
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
import time


def extract_zipfiles(src, dest, re_extract=False):
    if re_extract:
        if os.path.exists(dest):
            shutil.rmtree(dest)
    if not os.path.exists(dest):
        zipfile.ZipFile(src).extractall(dest)


def run_zipped_storylines(name, description_mult, zfile_path, number, choice_seed, pgr_seed, mode_sites, re_run,
                          run_pgr, run_export, plot, run_mp,
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
                save_daily_mult=True,
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
        outputs_dir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', name)
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
                        temp_data.loc[:, 'month'].str.split('-').str[1]).values)  # todo check

        outdata = pd.DataFrame(columns=columns)
        for mode, site in mode_sites:
            temp_data = pd.read_csv(os.path.join(outputs_dir, 'raw', f'{site}-{mode}', 'm_PGR.csv'),
                                    skiprows=1)  # just to get index
            temp_data.index = (temp_data.loc[:, 'month'].str.split('-').str[0] +
                               '-' +
                               temp_data.loc[:, 'month'].str.split('-').str[1]).values
            for m in plot_months:
                outdata.loc[:, f'{site}-{mode}_pg_m{m:02d}'] = temp_data.loc[:, m] * month_len
        outdata = corr_pg(outdata, mode_site=mode_sites)  # this calcs the 1 year
        outdata.to_csv(os.path.join(outputs_dir, 'corrected_data.csv'))
        outdata.mean().to_csv(os.path.join(outputs_dir, 'mean_corrected_data.csv'))

        # todo normalize to actual data to manage the seed changes

    if plot:
        data = pd.read_csv(os.path.join(outputs_dir, 'corrected_data.csv'), index_col=0)
        # todo plot!
        # todo overlayed mean plot for each site (pg)
        # todo overlayed mean with percentiles plot for each site (pg)
        # todo individaul pg violin plots
        # todo nested box plots per site
        raise NotImplementedError


def most_probabable(run_pgr, run_export, plot):
    t = time.time()
    number = 8121  # 10% of c. 81,000
    nsims = 100
    name = 'storage_most_probable'
    description_mult = 'the most probable/baseline for the storage runs'
    zip_path = os.path.join(ksl_env.proj_root, 'Storylines/final_storylines/Final_most_probable.zip')

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
        plot=plot,
        run_mp=True,
        start_run=0, nsims=nsims)

    print(f'{name} took {(time.time() - t) / 60} min to run {number} parallel with 10 sims for '
          f'{len(default_mode_sites)} mode-sites')


def scare(run_pgr, run_export, plot):
    t = time.time()
    number = 'all'  # 103 stories
    nsims = 100
    name = 'storage_scare'
    description_mult = 'scare scenarios with all storage systems'
    zip_path = os.path.join(ksl_env.proj_root,
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
        plot=plot,
        run_mp=True,
        start_run=0, nsims=nsims)

    print(f'{name} took {(time.time() - t) / 60} min to run {number} parallel with 10 sims for '
          f'{len(default_mode_sites)} mode-sites')


def hurt(run_pgr, run_export, plot):
    t = time.time()
    number = 'all'  # 226 stories
    nsims = 100
    name = 'storage_hurt'
    description_mult = 'hurt scenarios with all storage systems'
    zip_path = os.path.join(ksl_env.proj_root,
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
        plot=plot,
        run_mp=True,
        start_run=0, nsims=nsims)

    print(f'{name} took {(time.time() - t) / 60} min to run {number} parallel with 10 sims for '
          f'{len(default_mode_sites)} mode-sites')


if __name__ == '__main__':
    pgr = True
    export = False
    plot = False

    scare(pgr, export, plot)
    hurt(pgr, export, plot)
    most_probabable(pgr, export, plot)
