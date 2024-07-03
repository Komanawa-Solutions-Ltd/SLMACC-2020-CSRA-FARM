"""
 Author: Matt Hanson
 Created: 18/12/2020 11:23 AM
 """

import pandas as pd
import numpy as np
import os
import psutil
import netCDF4 as nc
import ksl_env
import datetime
import time
from BS_work.SWG.SWG_wrapper import measures_cor
from Climate_Shocks.Stochastic_Weather_Generator.irrigation_generator import get_irrigation_generator
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather, default_mode_sites, abs_max_irr
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth
from Climate_Shocks.Stochastic_Weather_Generator.read_swg_data import change_swg_units
from Climate_Shocks import climate_shocks_env
from Pasture_Growth_Modelling.historical_average_baseline import get_historical_average_baseline

# add basgra nz functions
from komanawa.basgra_nz_py.basgra_python import run_basgra_nz, get_month_day_to_nonleap_doy
from komanawa.basgra_nz_py.supporting_functions.output_metadata import get_output_metadata

# consider multiprocessing here???? no up a level (e.g. at teh storyline level)
default_pasture_growth_dir = os.path.join(os.path.join(ksl_env.slmmac_dir_unbacked, 'pasture_growth_sims'))
if not os.path.exists(default_pasture_growth_dir):
    os.makedirs(default_pasture_growth_dir)

out_metadata = get_output_metadata()

add_variables = {  # varaibles that are defined here and not in BASGRA
    'PGR': {'unit': 'kg dry matter/m2/day', 'description': 'pasture growth rate, calculated from yield'},
    'PGRA': {'unit': '\u0394 kg dry matter/m2/day', 'description': 'pasture growth rate anomaly from baseline'},
    'PGRA_CUM': {'unit': '\u0394 kg dry matter/m2 to date',
                 'description': 'cumulative pasture growth rate anomaly from baseline'},
    'PER_PAW': {'unit': 'fraction', 'description': 'fraction of PAW (profile available water'},
    'F_REST': {'unit': 'fraction', 'description': 'fraction of irrigation restriction, '
                                                  '1=0 mm water available/day, '
                                                  '0 = {} mm water available/day,'.format(abs_max_irr)}
}

out_metadata.update(add_variables)

memory_per_run = (140.3 * 1.049e+6) / 2 / 100 * 1.1  # 140 mib for 100 2 year sims and add 10% slack so c. 2.3mb/3yrsim

out_variables = (
    'BASAL',  # should some of these be amalgamated to sum, no you can multiply against # of days in the month.
    'PGR',
    'PER_PAW',
    'F_REST',
    'IRRIG',
    'IRRIG_DEM',
    'RESEEDED',
    'DMH_RYE',
    'DMH_WEED',
    'YIELD',
    'DRAIN',
    'irrig_dem_store',
    'irrig_store',
    'irrig_scheme',
    'h2o_store_vol',
    'h2o_store_per_area',
    'IRR_TRIG_store',
    'IRR_TARG_store',
    'store_runoff_in',
    'store_leak_out',
    'store_irr_loss',
    'store_evap_out',
    'store_scheme_in',
    'store_scheme_in_loss',

)

from socket import gethostname

if gethostname() == 'wanganui':
    irr_gen = None
else:
    irr_gen = get_irrigation_generator()

month_len = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

default_swg_dir = os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', 'full_SWG')


def run_pasture_growth(storyline_path, outdir, nsims, mode_sites=default_mode_sites, padock_rest=False,
                       save_daily=False, description='', swg_dir=default_swg_dir, verbose=True,
                       n_parallel=1, fix_leap=True, re_run=True, seed=None, use_1_seed=False,
                       use_out_variables=out_variables):
    """
    creates weather data, runs basgra and saves values to a netcdf
    :param storyline_path: path to the storyline
    :param outdir: directory to save the files to
    :param nsims: number of simulations to run
    :param mode_sites: a list of modes and sites some or all of:     ('dryland', 'oxford'),
                                                                     ('irrigated', 'eyrewell'),
                                                                     ('irrigated', 'oxford'),
    :param padock_rest: boolean if True run paddock restrictions,
    :param save_daily: boolean if True save daily data to netcdf otherwise just save monthly
    :param description: description to append to the default description for the netcdf
    :param swg_dir: directory to the stocastic weather dir which has full netcdfs for all possible weather states
    :param verbose: boolean, if True then print data as it runs
    :param n_parallel: int, the number of parallel runs happening (used to prevent memory errors from large sims while
                       multiprocessing.  if calling this funtion, then simply leave as 1
    :param seed: allows sudo random to be reproducable
    :param use_1_seed: bool if True use the same seed for each month of the storyline, if false calculate a new seed
                       for each month of each storyline.  The new seeds are based on the passed seed so they are
                       reproducable (via seed kwarg); however each month will have different data which can cause.
                       problems if comparing two datasets of differnt length (e.g. 3 year run or 3 years run separately
                       if comparing between site/modes for the same storyline then use_1_seed should be false
    :return: None
    """
    t = time.time()
    assert isinstance(n_parallel, int)
    assert n_parallel > 0
    zipped = False
    try:
        storyline_key = os.path.splitext(os.path.basename(storyline_path))[0]
    except TypeError:
        storyline_key = os.path.splitext(os.path.basename(storyline_path.name))[0]  # to allow zipped files
        zipped = True

    if zipped:
        storyline_path.seek(0)
    storyline = pd.read_csv(storyline_path)
    simlen = np.array([month_len[e] for e in storyline.month]).sum()

    sm = storyline.month.iloc[0]
    sy = storyline.year.iloc[0]
    ey = storyline.year.iloc[-1]
    em = storyline.month.iloc[-1]
    out_index = pd.date_range(f'{sy}-{sm:02d}-1', f'{ey}-{em:02d}-{month_len[em]}')
    if fix_leap:
        out_index = out_index[~((out_index.month == 2) & (out_index.day == 29))]
        out_index = pd.to_datetime(
            [f'{y}-{m:02d}-{d:02d}' for y, m, d in zip(out_index.year, out_index.month, out_index.day)])
    assert simlen == len(out_index), f'simlen should be {simlen}, but is {len(out_index)}, check for leap years'

    if zipped:
        storyline_path.seek(0)
        storyline_text = storyline_path.readlines()
    else:
        with open(storyline_path, 'r') as f:
            storyline_text = f.readlines()

    assert isinstance(nsims, int), 'nsims must be an integer instead {}: {}'.format(nsims, type(nsims))
    for mode, site in mode_sites:
        outpath = os.path.join(outdir, '{}-{}-{}.nc'.format(storyline_key, site, mode))
        if not re_run and os.path.exists(outpath):
            print(f'skipping {os.path.basename(outpath)}')
            continue

        _run_simple_rest(storyline=storyline, nsims=nsims, mode=mode, site=site, simlen=simlen,
                         storyline_key=storyline_key,
                         outdir=outdir,
                         save_daily=save_daily, description=description, storyline_text=storyline_text, swg_dir=swg_dir,
                         verbose=verbose, n_parallel=n_parallel, fix_leap=fix_leap, seed=seed, use_1_seed=use_1_seed,
                         use_out_variables=use_out_variables)
        if padock_rest:
            _run_paddock_rest(storyline_key=storyline_key, outdir=outdir, storyline=storyline, nsims=nsims, mode=mode,
                              site=site, simlen=simlen,
                              save_daily=save_daily, description=description, storyline_text=storyline_text,
                              swg_dir=swg_dir, verbose=verbose, n_parallel=n_parallel, fix_leap=fix_leap, seed=seed,
                              use_1_seed=use_1_seed, use_out_variables=use_out_variables)
    t = time.time() - t
    if verbose:
        print(f'took {t / 60} min to run {nsims} sims paddock_rest{padock_rest}')


def get_rest_tolerance(r):
    return max([0.02, 0.1 * r])


def get_irr_data(num_to_pull, storyline, simlen, seed=None, use_1_seed=False):
    if use_1_seed:
        seeds = np.repeat(seed, len(storyline) * 10)  # to allow linked comparison
    else:
        np.random.seed(seed)
        seeds = np.random.randint(0, 3548324, len(storyline) * 10)  # make too many
    out = np.zeros((num_to_pull, simlen))
    idx = 0
    for i, (m, pstate, r) in enumerate(storyline.loc[:, ['month', 'precip_class', 'rest']].itertuples(False, None)):
        if m in [6, 7, 8]:  # skip months without irrigation
            idx += month_len[m]
            continue
        if pstate == 'D':
            plet = 'D'
        else:
            plet = 'ND'
        key = 'm{:02d}-{}'.format(m, plet)
        out[:, idx: idx + month_len[m]] = irr_gen.get_data(num_to_pull, key=key, suffix_selection=r,
                                                           tolerance=get_rest_tolerance(r),
                                                           max_replacement_level=0.1,
                                                           under_level='warn',
                                                           seed=seeds[i])
        idx += month_len[m]

    return out


def _gen_input(storyline, nsims, mode, site, chunks, current_c, nperc, simlen, swg_dir, fix_leap,
               seed=None, use_1_seed=False):
    """

    :param storyline: loaded storyline
    :param SWG_path: path to the directory with contining the files from the SWG
    :param nsims: number of sims to run
    :param mode: one of ['irrigated', 'dryland']
    :param site: one of ['eyrewell', 'oxford']
    :param chunks: the number of chunks
    :param current_c: the current chunk (from range(chunks)
    :param nperc: number of simulations that can be run per chunk
    :param use_1_seed: all of the seeds for different months will be identical. ,
                       useful for comparison of linked - non linked
    :return:
    """
    # manage seeds
    np.random.seed(seed)
    seeds = np.random.randint(0, 1568254, 25)
    iseed = 0

    # manage chunks
    if chunks == 1:
        num_to_pull = nsims
    elif chunks > 1:
        num_to_pull = nperc
        if current_c + 1 == chunks:
            # manage last chunk
            num_to_pull = nsims - (current_c * nperc)
    else:
        raise ValueError('shouldnt get here')

    params, doy_irr = get_params_doy_irr(mode, site)
    matrix_weathers = []
    days_harvests = []

    # get restriction data
    if mode == 'dryland':
        rest_data = np.repeat([None], num_to_pull)
    elif mode == 'irrigated' or 'store' in mode:
        rest_data = get_irr_data(num_to_pull, storyline, simlen, seed=seeds[iseed], use_1_seed=use_1_seed)
        iseed += 1
    else:
        raise ValueError('weird arg for mode: {}'.format(mode))
    # get weather data
    weather_data = _get_weather_data(storyline=storyline, nsims=num_to_pull, simlen=simlen, swg_dir=swg_dir, site=site,
                                     fix_leap=fix_leap, seed=seeds[iseed], use_1_seed=use_1_seed)
    iseed += 1

    # make all the other data
    for rest, weather in zip(rest_data, weather_data):
        if rest is None:
            rest_temp = None
        else:
            rest_temp = pd.DataFrame(data=rest, index=weather.index, columns=['frest'])
        matrix_weather = create_matrix_weather(mode, weather_data=weather, restriction_data=rest_temp,
                                               rest_key='frest', fix_leap=fix_leap)
        matrix_weathers.append(matrix_weather)
        days_harvests.append(create_days_harvest(mode, matrix_weather, site, fix_leap=fix_leap))

    return params, doy_irr, matrix_weathers, days_harvests


def _run_simple_rest(storyline, nsims, mode, site, simlen, storyline_key, outdir,
                     save_daily, description, storyline_text, swg_dir, verbose, n_parallel, fix_leap, seed,
                     use_1_seed, use_out_variables):
    number_run = int(
        psutil.virtual_memory().available // memory_per_run * (simlen / 365) / n_parallel
    )
    chunks = int(-1 * (-nsims // number_run))
    if chunks == 1:
        number_run = nsims
    all_outpaths = []
    for c in range(chunks):
        if verbose:
            print('starting chunk {} of {} for simple irrigation restrictions {}-{}'.format(c + 1, chunks, site, mode))
        params, doy_irr, all_matrix_weathers, all_days_harvests = _gen_input(storyline=storyline,
                                                                             nsims=nsims, mode=mode, site=site,
                                                                             chunks=chunks, current_c=c,
                                                                             nperc=number_run, simlen=simlen,
                                                                             swg_dir=swg_dir, fix_leap=fix_leap,
                                                                             seed=seed, use_1_seed=use_1_seed)

        all_out = np.zeros((len(use_out_variables), simlen, number_run)) * np.nan
        for i, (matrix_weather, days_harvest) in enumerate(zip(all_matrix_weathers, all_days_harvests)):
            restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
            out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False, run_365_calendar=fix_leap)
            out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

            pg = pd.DataFrame(
                calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
            out.loc[:, 'PGR'] = pg.loc[:, 'pg']
            out.loc[:, 'F_REST'] = restrict

            all_out[:, :, i] = out.loc[:, use_out_variables].values.transpose()

        # one netcdf file for each mode/site
        month = out.index.month.values
        doy = out.loc[:, 'doy'].values
        year = out.loc[:, 'year'].values
        outpath = _output_to_nc(storyline_key=storyline_key, storyline_text=storyline_text, outdir=outdir,
                                outdata=all_out,
                                nsims=nsims,
                                month=month, doy=doy, year=year,
                                chunks=chunks, current_c=c, nperchunk=number_run, save_daily=save_daily,
                                description=description, site=site, mode=mode, verbose=verbose,
                                use_out_variables=use_out_variables)

    add_pasture_growth_anaomoly_to_nc(outpath)


def _run_paddock_rest(storyline_key, outdir, storyline, nsims, mode, site, simlen,
                      save_daily, description, storyline_text, swg_dir, verbose, n_parallel, fix_leap, seed,
                      use_1_seed, use_out_variables):
    """
    run storyline through paddock restrictions...
    :param storyline:
    :param swg_path:
    :param nsims:
    :param mode:
    :param site:
    :return:
    """
    # paddock level restrictions
    levels = np.arange(0, 125, 25) / 100  # levels already capture the extra (mean run) as levels has start stop
    number_run = int(
        psutil.virtual_memory().available // (memory_per_run * (simlen / 365) * len(levels)) / n_parallel
    )
    chunks = int(-1 * (-nsims // number_run))
    if chunks == 1:
        number_run = nsims

    all_outpaths = []
    for c in range(chunks):
        if verbose:
            print('running paddock pasture growth for chunk {} of {} {}-{}'.format(c + 1, chunks, site, mode))
        params, doy_irr, all_matrix_weathers, all_days_harvests = _gen_input(storyline=storyline,
                                                                             nsims=nsims, mode=mode, site=site,
                                                                             chunks=chunks, current_c=c,
                                                                             nperc=number_run, simlen=simlen,
                                                                             swg_dir=swg_dir, fix_leap=fix_leap,
                                                                             seed=seed, use_1_seed=use_1_seed)

        all_out = np.zeros((len(use_out_variables), simlen, len(levels) - 1, number_run)) * np.nan
        out_names = []
        out_limits = []
        for j, (matrix_weather, days_harvest) in enumerate(zip(all_matrix_weathers, all_days_harvests)):
            restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
            matrix_weather.loc[:, 'max_irr'] = abs_max_irr
            for i, (ll, lu) in enumerate(zip(levels[0:-1], levels[1:])):
                if j == 0:
                    out_name = 'paddock_{}'.format(i)
                    out_names.append(out_name)
                    out_limits.append('lower_{}_upper_{}'.format(ll, lu))
                matrix_weather_new = matrix_weather.copy(deep=True)

                matrix_weather_new.loc[restrict <= ll, 'max_irr'] = abs_max_irr
                matrix_weather_new.loc[restrict >= lu, 'max_irr'] = 0
                idx = (restrict > ll) & (restrict < lu)
                matrix_weather_new.loc[idx, 'max_irr'] = abs_max_irr * ((restrict.loc[idx] - ll) / 0.25)

                temp = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False,
                                     run_365_calendar=fix_leap)
                temp.loc[:, 'PER_PAW'] = temp.loc[:, 'PAW'] / temp.loc[:, 'MXPAW']
                temp.loc[:, 'PGR'] = calc_pasture_growth(temp, days_harvest, 'from_yield', '1D', resamp_fun='mean')
                temp.loc[:, 'F_REST'] = restrict

                all_out[:, :, i, j] = temp.loc[:, use_out_variables].values.transpose()
        # one netcdf for each mode,site, (paddock, mean)
        month = temp.index.month.values
        doy = temp.loc[:, 'doy'].values
        year = temp.loc[:, 'year'].values
        outpaths = _output_to_nc_paddock(storyline_key=storyline_key, storyline_text=storyline_text, outdir=outdir,
                                         outdata=all_out, nsims=nsims,
                                         month=month, doy=doy, year=year,
                                         chunks=chunks, current_c=c, nperchunk=number_run,
                                         out_names=out_names, out_limits=out_limits,
                                         save_daily=save_daily, description=description, site=site, mode=mode,
                                         verbose=verbose, use_out_variables=use_out_variables)
        all_outpaths.extend(outpaths)

    for p in np.unique(all_outpaths):
        add_pasture_growth_anaomoly_to_nc(p)


def _output_to_nc(storyline_key, storyline_text, outdir, outdata, nsims, month, doy, year,
                  chunks, current_c, nperchunk, site, mode, use_out_variables, save_daily=False, description='',
                  verbose=True):
    """

    :param storyline_key: key to the storyline
    :param outdir: directory to save
    :param outdata: data from non paddock run
    :param nsims: total number of sims
    :param month: month data from run
    :param doy: doy data from run
    :param year: year data from run
    :param chunks: number of chunks
    :param current_c: current chunks
    :param nperchunk: number of simulations per chunk
    :param save_daily: boolean if True save daily data as well as monthly mean
    :param description: additional description
    :return:
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, '{}-{}-{}.nc'.format(storyline_key, site, mode))

    # get netcdf file instance
    if current_c == 0:
        nc_file = _create_nc_file(outpath, number_run=nsims, month=month, doy=doy, year=year,
                                  storyline_text=storyline_text,
                                  save_daily=save_daily, description=description, verbose=verbose,
                                  use_out_variables=use_out_variables)
    else:
        nc_file = nc.Dataset(outpath, 'a')

    _write_data(nc_file, year, month, outdata, current_c, nperchunk, chunks, nsims, save_daily, verbose=verbose,
                use_out_variables=use_out_variables)
    nc_file.close()
    return outpath


def _output_to_nc_paddock(storyline_key, storyline_text, outdir, outdata, nsims, month, doy, year,
                          chunks, current_c, nperchunk,
                          out_names, out_limits, site, mode, use_out_variables,
                          save_daily=False, description='', verbose=True):
    """

    :param storyline_key: key to the storyline
    :param outdir: directory to save
    :param outdata: data from non paddock run
    :param nsims: total number of sims
    :param month: month data from run
    :param doy: doy data from run
    :param year: year data from run
    :param chunks: number of chunks
    :param current_c: current chunks
    :param nperchunk: number of simulations per chunk
    :param out_names: string of the out name
    :param out_limits string of the out limits
    :param save_daily: boolean if True save daily data as well as monthly mean
    :param description: additional description
    :return:
    """
    out_paths = []
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # outdata shape np.zeros((len(out_variables), simlen, len(levels) - 1, number_run))
    for i, (outn, outl) in enumerate(zip(out_names, out_limits)):
        outpath = os.path.join(outdir, '{}-{}-{}-{}.nc'.format(storyline_key, outn, site, mode, ))
        extra = ('\npaddock restrictions are broken into {} different paddocks, where '.format(len(out_names)) +
                 'each paddock absorbs specific restrictions defined by limts:\n')
        description = description + extra + '\n' + outn + '\n' + outl

        # get netcdf file instance
        if current_c == 0:
            nc_file = _create_nc_file(outpath, number_run=nsims, month=month, doy=doy, year=year,
                                      storyline_text=storyline_text,
                                      save_daily=save_daily, description=description, verbose=verbose,
                                      use_out_variables=use_out_variables)
        else:
            nc_file = nc.Dataset(outpath, 'a')

        _write_data(nc_file, year, month, outdata[:, :, i, :], current_c, nperchunk, chunks, nsims, save_daily,
                    verbose=verbose, use_out_variables=use_out_variables)
        nc_file.close()
        out_paths.append(outpath)

    # save the mean data
    outpath = os.path.join(outdir, '{}-paddock-mean-{}-{}.nc'.format(storyline_key, site, mode))
    extra = ('\npaddock restrictions are broken into {} different paddocks, where '.format(len(out_names)) +
             'each paddock absorbs specific restrictions defined by limts this file represents the mean of all'
             'paddocks')
    description = description + extra + '\n'

    # get netcdf file instance
    if current_c == 0:
        nc_file = _create_nc_file(outpath, number_run=nsims, month=month, doy=doy, year=year,
                                  storyline_text=storyline_text,
                                  save_daily=save_daily, description=description, verbose=verbose,
                                  use_out_variables=use_out_variables)
    else:
        nc_file = nc.Dataset(outpath, 'a')

    outdata = np.nanmean(outdata, axis=2)
    _write_data(nc_file, year, month, outdata, current_c, nperchunk, chunks, nsims, save_daily, verbose=verbose,
                use_out_variables=use_out_variables)
    out_paths.append(outpath)
    nc_file.close()
    return out_paths


def _write_data(nc_file, year, month, outdata, current_c, nperchunk, chunks, nsims, save_daily,
                verbose, use_out_variables):
    if verbose:
        print(' writing data for {}'.format(nc_file.filepath()))
    # write monthly data chunk
    for i, v in enumerate(use_out_variables):
        temp = nc_file.variables['m_{}'.format(v)]
        # outdata shape = np.zeros((len(out_variables), simlen_monthly, number_run))
        to_val = (current_c + 1) * nperchunk
        # capture last chunk...
        if current_c + 1 == chunks:
            to_val = nsims

        # resample to monthly data
        yearmonths = list(pd.Series(zip(year, month)).unique())
        resamp_data = np.zeros((len(yearmonths), to_val - current_c * nperchunk)) * np.nan
        for k, (y, m) in enumerate(yearmonths):
            idx = ((year == y) & (month == m))
            if v == 'RESEEDED':  # make reseed easier to understand (1 if reseeds in the month)
                resamp_data[k, :] = np.nansum(outdata[i, idx, :], axis=0)[:to_val - current_c * nperchunk]
            else:
                resamp_data[k, :] = np.nanmean(outdata[i, idx, :], axis=0)[:to_val - current_c * nperchunk]

        temp[:, (current_c * nperchunk): to_val] = resamp_data

        # write daily data chunk
    if save_daily:
        for i, v in enumerate(use_out_variables):
            temp = nc_file.variables['d_{}'.format(v)]
            to_val = (current_c + 1) * nperchunk
            # capture last chunk...
            if current_c + 1 == chunks:
                to_val = nsims

            # outdata shape = np.zeros((len(out_variables), simlen, number_run))
            temp[:, (current_c * nperchunk): to_val] = outdata[i, :, :][:, :to_val - current_c * nperchunk]


def _create_nc_file(outpath, number_run, month, doy, year, storyline_text, use_out_variables,
                    save_daily=False, description='',
                    verbose=True):
    """
    create the output netcdf files, create variables and populate metadata
    :param outpath: path to the file (.nc)
    :param number_run: number of simulations
    :param month: daily month data (output of basgra)
    :param doy: daily doy data (output of basgra)
    :param year: daily year data (output of basgra)
    :param storyline_text: the csv file of the storyline loaded as text
    :param save_daily: boolean, if True save raw daily data as well as the monthly mean.  if False save only the
                       monthly mean
    :param description: an additional description to add to the netcdf description
    :return: nc.Dataset(outpath,w)
    """
    if verbose:
        print(' creating nc file {}'.format(outpath))
    nc_file = nc.Dataset(outpath, 'w')

    # set global attributes
    t = ''
    if save_daily:
        t = 'and the raw daily data'
    description = ('this file contains the monthly mean {} '.format(t) +
                   'outputs of the BASGRA pasture Growth model run on data ' +
                   'created by the Stocasitc weather and restriction generators for the SLMMAC-2020-CSRA\n\n' +
                   'the storyline for this run is defined in the .storyline attribute\n\n' +
                   'the variables have either "d_" or "m_" as a prefix, this denotes whether the data is the ' +
                   'raw daily simmulation data or  monthly mean value note that the daily data may not have been' +
                   'saved here\n\n in addition there are the following additional ' +
                   'description: {}'.format(description)
                   )
    nc_file.basgra = ('version: {}, ' +
                      'https://github.com/Komanawa-Solutions-Ltd/BASGRA_NZ_PY').format(ksl_env.basgra_version)
    nc_file.description = description
    nc_file.history = 'created {}'.format(datetime.datetime.now().isoformat())
    nc_file.source = 'script: {}'.format(__file__)
    nc_file.contacts = 'Matt Hanson: matt@komanawa.com\n Zeb Etheridge: zeb@komanawa.com'
    nc_file.storyline = storyline_text

    # calculate monthly years and months
    temp = list(pd.Series(zip(year, month)).unique())
    m_months = [e[1] for e in temp]
    m_years = [e[0] for e in temp]

    # make dimensions and dimension variables
    nc_file.createDimension('sim_month', len(m_months))
    nc_file.createDimension('realisation', number_run)

    if save_daily:
        nc_file.createDimension('sim_day', len(doy))

    # make dimension variables
    real = nc_file.createVariable('real', int, ('realisation',))
    real.setncatts({'units': 'none',
                    'long_name': 'realisation number',
                    'missing_value': -1})
    real[:] = np.arange(number_run)

    my = nc_file.createVariable('m_year', int, ('sim_month',))
    my.setncatts({'units': 'none',
                  'long_name': 'year of simulation for the monthly data',
                  'comments': ('the start year was set as 2025 to allow a 3 year stretch of non-leap years the '
                               'simulations were created to look at "now" or between 2015-2025'),
                  'missing_value': -1})

    my[:] = m_years

    mm = nc_file.createVariable('m_month', int, ('sim_month',))
    mm.setncatts({'units': 'none',
                  'long_name': 'calander month for the monthly data',
                  'missing_value': -1})
    mm[:] = m_months

    if save_daily:
        dy = nc_file.createVariable('d_year', int, ('sim_day',))
        dy.setncatts({'units': 'none',
                      'long_name': 'year of simulation for the daily data',
                      'comments': ('the start year was set as 2025 to allow a 3 year stretch of non-leap years the '
                                   'simulations were created to look at "now" or between 2010-2030'),
                      'missing_value': -1})
        dy[:] = year

        ddoy = nc_file.createVariable('d_doy', int, ('sim_day',))
        ddoy.setncatts({'units': 'none',
                        'long_name': 'day of the year for the daily data',
                        'missing_value': -1})
        ddoy[:] = doy

    # make output variables
    for v in use_out_variables:
        ex = ''
        if v == 'RESEEDED':
            ex = (' for RESEEDED the monthly data is summed across the month, so if any reseed event happened '
                  'it will be 1 as only 1 reseed event is allowed per year in these sims')
        temp = nc_file.createVariable('m_{}'.format(v), float, ('sim_month', 'realisation'))
        temp.setncatts({'units': out_metadata[v.upper()]['unit'],
                        'description': out_metadata[v.upper()]['description'] + ex,
                        'long_name': 'monthly mean {}'.format(v),
                        'missing_value': np.nan})

        if save_daily:
            temp2 = nc_file.createVariable('d_{}'.format(v), float, ('sim_day', 'realisation'))
            temp2.setncatts({'units': out_metadata[v.upper()]['unit'],
                             'description': out_metadata[v.upper()]['description'],
                             'long_name': 'daily raw {}'.format(v),
                             'missing_value': np.nan})

    return nc_file


def _get_weather_data(storyline, nsims, simlen, swg_dir, site, fix_leap, seed=None, use_1_seed=False):
    if use_1_seed:
        seeds = np.repeat(seed, len(storyline) * 10)  # to alow direct comparison between linked and non-linked data
    else:
        np.random.seed(seed)
        seeds = np.random.randint(0, 3548324, len(storyline) * 10)  # make too many
    iseed = 0

    out_array = np.zeros((simlen, len(measures_cor), nsims))
    i = 0
    if site == 'eyrewell':
        var = 'main_site'
    elif site == 'oxford':
        var = 'exsites_P0'
    else:
        raise ValueError('incorrect site {} expected one of [eyrewell, oxford]')

    sm = storyline.month.iloc[0]
    sy = storyline.year.iloc[0]
    ey = storyline.year.iloc[-1]
    em = storyline.month.iloc[-1]
    out_index = pd.date_range(f'{sy}-{sm:02d}-1',
                              f'{ey}-{em:02d}-{month_len[em]}')

    if fix_leap:
        out_index = out_index[~((out_index.month == 2) & (out_index.day == 29))]
        out_index = pd.to_datetime(
            [f'{y}-{m:02d}-{d:02d}' for y, m, d in zip(out_index.year, out_index.month, out_index.day)])

    for p, t, m in storyline.loc[:, ['precip_class', 'temp_class', 'month']].itertuples(False, None):
        temp = nc.Dataset(os.path.join(swg_dir, 'm{m:02d}-{t}-{p}-0_all.nc'.format(m=int(m), t=t, p=p)))
        np.random.seed(seeds[iseed])
        iseed += 1
        idxs = np.random.randint(temp.dimensions['real'].size, size=(nsims,))

        out_array[i: i + month_len[m], :, :] = np.array(temp.variables[var][:, :, idxs])
        i += month_len[m]
        temp.close()
    outdata = []
    mapper = get_month_day_to_nonleap_doy(key_doy=False)
    for s in range(nsims):
        t = pd.DataFrame(out_array[:, :, s], columns=measures_cor, index=out_index)
        t.index.name = 'date'
        t.loc[:, 'year'] = t.index.year
        t.loc[:, 'doy'] = [mapper[(m, d)] for m, d in zip(t.index.month, t.index.day)]
        t.loc[:, 'month'] = t.index.month
        change_swg_units(t)
        outdata.append(t)
    # output is list of pd dataframes that can go into matrix weather
    return outdata


def add_pasture_growth_anaomoly_to_nc(nc_path):
    """
    add the pasture growth anaomaly from basline sim where sim is greater in length than baseline, simply use the last
    year of the baseline data to compare.  if the sim has leap days in it, then shift data and backfill data.
    :param nc_path:
    :param recalc:
    :return:
    """
    data = nc.Dataset(nc_path, 'a')
    site, mode = os.path.basename(nc_path).strip('.nc').split('-')[-2:]
    # monthly
    sim_months = np.array(data.variables['m_month'])
    sim_years = np.array(data.variables['m_year'])
    base_data, storyline_text, base_run_date, story_nm = _get_baseline_pgr(site, mode, sim_months, sim_years, False)
    sim_data = np.array(data.variables['m_PGR'])
    if 'm_PGRA' not in data.variables.keys():  # needed to allow re-write
        pga_an = data.createVariable('m_PGRA', float, ('sim_month', 'realisation'), fill_value=np.nan)
    else:
        pga_an = data.variables['m_PGRA']
    pga_an.setncatts({'units': 'kgDM/ha/day',
                      'base_storyline_text': storyline_text,
                      'base_run_date': base_run_date,
                      'long_name': f'monthly pasture growth anomaly from {story_nm}',
                      'missing_value': np.nan,
                      'direction': 'sim - base'})
    pga_an[:] = sim_data - base_data[:, np.newaxis]

    # cumulative
    if 'm_PGRA_cum' not in data.variables.keys():
        pga_an = data.createVariable('m_PGRA_cum', float, ('sim_month', 'realisation'), fill_value=np.nan)
    else:
        pga_an = data.variables['m_PGRA_cum']
    pga_an.setncatts({'units': 'kgDM/ha/day',
                      'base_storyline_name': story_nm,
                      'base_storyline_text': storyline_text,
                      'base_run_date': base_run_date,
                      'long_name': f'cumulative monthly pasture growth anomaly from {story_nm}, adj for ndays in month',
                      'missing_value': np.nan,
                      'direction': 'sim - base'})
    pga_an[:] = np.nancumsum((sim_data - base_data[:, np.newaxis]) *
                             np.array([month_len[e] for e in sim_months])[:, np.newaxis]
                             , axis=0)

    # daily
    if 'd_year' in data.variables.keys():
        sim_days = np.array(data.variables['d_doy'])
        sim_years = np.array(data.variables['d_year'])
        sim_data = np.array(data.variables['d_PGR'])

        base_data, storyline_text, base_run_date, story_nm = _get_baseline_pgr(site, mode, sim_days, sim_years, True)
        if 'd_PGRA' not in data.variables.keys():
            pga_an = data.createVariable('d_PGRA', float, ('sim_day', 'realisation'), fill_value=np.nan)
        else:
            pga_an = data.variables['d_PGRA']
        pga_an.setncatts({'units': 'kgDM/ha/day',
                          'base_storyline_name': story_nm,
                          'base_storyline_text': storyline_text,
                          'base_run_date': base_run_date,
                          'long_name': f'daily pasture growth anomaly from {story_nm}',
                          'missing_value': np.nan,
                          'direction': 'sim - base'})
        pga_an[:] = sim_data - base_data[:, np.newaxis]

        # cumulative
        if 'd_PGRA_cum' not in data.variables.keys():
            pga_an = data.createVariable('d_PGRA_cum', float, ('sim_day', 'realisation'), fill_value=np.nan)
        else:
            pga_an = data.variables['d_PGRA_cum']
        pga_an.setncatts({'units': 'kgDM/ha/day',
                          'base_storyline_name': story_nm,
                          'base_storyline_text': storyline_text,
                          'base_run_date': base_run_date,
                          'long_name': f'cumulative daily pasture growth anomaly from {story_nm}',
                          'missing_value': np.nan,
                          'direction': 'sim - base'})
        pga_an[:] = np.nancumsum(sim_data - base_data[:, np.newaxis], axis=0)

    data.close()


def _get_baseline_pgr(site, mode, sim_mon_day, sim_years, daily):
    """

    :param site:
    :param mode:
    :param daily:
    :param recalc: normal reacl
    :return: baseline name, baseline data
    """
    if daily:
        sim_mon_day_nm = 'doy'
    else:
        sim_mon_day_nm = 'month'

    story_nm = 'historical average 1972 - 2019'
    base_sim_path = os.path.join(default_pasture_growth_dir, "baseline_sim_no_pad", f"0-baseline-{site}-{mode}.nc")
    storyline_text = 'historical average from 1972-2019 averaged by month and then re-sampled to daily data'

    # read in the historical average
    base_data, base_run_date = get_historical_average_baseline(site, mode, years=np.unique(sim_years))

    # get the base data aligned to the simdata (e.g. don't assume same time steps)
    # manage longer data than baseline sim
    max_year = sim_years.max()
    base_years_max = base_data.year.max()
    base_data = base_data.set_index(['year', sim_mon_day_nm])

    if max_year > base_years_max:
        # set up expected data
        fill_mon_day = base_data.index.levels[1]
        fill_year = []
        for v in fill_mon_day:
            try:
                base_data.loc[(base_years_max, v), 'PGR']
                ov = base_years_max
            except KeyError:
                ov = base_years_max - 1
            fill_year.append(ov)

        fill_data = base_data.loc[zip(fill_year, fill_mon_day), 'PGR'].values
        for y in range(0, max_year - base_years_max + 1):
            for fm, fd in zip(fill_mon_day, fill_data):
                base_data.loc[(base_years_max + y, fm), 'PGR'] = fd

    # handle sims with leap years when sim does not have leap day
    if daily and 366 in sim_mon_day:
        # data has leap years...
        mapper = get_month_day_to_nonleap_doy(True)
        temp = pd.to_datetime(
            [f'{y}-{mapper[doy][0]:02d}-{mapper[doy][1]:02d}' for y, doy in base_data.index.values])
        base_data = base_data.reset_index()
        base_data.loc[:, 'sim_mon_day'] = temp.dayofyear
        base_data_temp = base_data.set_index(['year', 'sim_mon_day'])

        base_data = pd.DataFrame(index=pd.date_range(temp.min(), temp.max()))
        base_data.loc[:, 'year'] = base_data.index.year
        base_data.loc[:, 'sim_mon_day'] = base_data.index.dayofyear
        base_data.set_index(['year', 'sim_mon_day'], inplace=True)
        base_data.loc[base_data_temp.index, 'PGR'] = base_data_temp.loc[:, 'PGR'].values
        base_data.fillna('bfill', inplace=True)

    base_data = base_data.loc[zip(sim_years, sim_mon_day), 'PGR'].reset_index().drop_duplicates().PGR.values
    assert sim_years.shape == base_data.shape
    return base_data, storyline_text, base_run_date, story_nm


def fix_reseeded(nc_path):
    data = nc.Dataset(nc_path, 'a')
    t = data.variables['m_RESEEDED']
    t[:] = (np.array(t) > 0).astype(float)
    data.close()


if __name__ == '__main__':
    add_pasture_growth_anaomoly_to_nc(
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\random_good_irr\rsl-000000-oxford-irrigated.nc")
