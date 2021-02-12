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
from Climate_Shocks.story_swg_iid_managment import storyline_swg_paths
from Climate_Shocks.Stochastic_Weather_Generator.read_swg_data import read_swg_data
from Climate_Shocks.Stochastic_Weather_Generator.irrigation_generator import get_irrigation_generator
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather, default_mode_sites, abs_max_irr
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from basgra_python import run_basgra_nz
from supporting_functions.output_metadata import get_output_metadata

# todo check that the SWG realisations fit our criterion!?! yes need to, but here or up a level at the creation of the realisations

# todo # consider multiprocessing here???? no up a level (e.g. at teh storyline level)

# todo check!

# todo check the amount of memory/time to run a full suite!

out_metadata = get_output_metadata()

memory_per_run = 1  # todo define in bytes

out_variables = (
    'BASAL',  # todo fill out
    'PGR',
)

irr_gen = get_irrigation_generator()


def get_rest_tolerance(r):
    return max([0.02, 0.1 * r])  # todo check if these data are avalible in the SWG...


def get_irr_data(num_to_pull, storyline_path, simlen):
    storyline = pd.read_csv(storyline_path)
    out = np.zeros((num_to_pull, simlen))
    for i, (m, pstate, r) in enumerate(storyline.loc[:, ['month', 'precip_class', 'rest']].itertuples(False, None)):
        if pstate >= 0.9:
            plet = 'D'
        else:
            plet = 'ND'
        key = 'm{:02d}-{}'.format(m, plet)
        out[:, irr_gen.sim_len[key]] = irr_gen.get_data(num_to_pull, key=key, mean=r, tolerance=get_rest_tolerance(r))

    return out


def _gen_input(storyline_path, SWG_path, nsims, mode, site, chunks, current_c, nperc, simlen):
    """

    :param storyline_path: path to the .csv storyline path
    :param SWG_path: path to the directory with contining the files from the SWG
    :param nsims: number of sims to run
    :param mode: one of ['irrigated', 'dryland']
    :param site: one of ['eyrewell', 'oxford']
    :param chunks: the number of chunks
    :param current_c: the current chunk (from range(chunks)
    :param nperc: number of simulations that can be run per chunk
    :return:
    """
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

    params, doy_irr = get_params_doy_irr(mode)
    matrix_weathers = []
    days_harvests = []

    # get restriction data
    if mode == 'dryland':
        rest_data = np.repeat([None], num_to_pull)
    elif mode == 'irrigated':
        rest_data = get_irr_data(num_to_pull, storyline_path, simlen)
    else:
        raise ValueError('weird arg for mode: {}'.format(mode))

    # get weather data
    swg_paths = pd.Series(sorted(os.listdir(SWG_path)))
    swg_paths = swg_paths.loc[swg_paths.str.contains('.nc')]  # get rid of any non-nc files like YML
    if site == 'eyrewell':
        exsite = False
        swg_paths = list(swg_paths.loc[~swg_paths.str.contains('exsites')])

    elif site == 'oxford':
        exsite = True
        swg_paths = list(swg_paths.loc[swg_paths.str.contains('exsites')])

    else:
        raise ValueError('weird input for site: {}'.format(site))
    weather_data = read_swg_data(swg_paths, exsite)

    # make all the other data
    for rest, weather in zip(rest_data, weather_data):
        rest_temp = pd.DataFrame(data=rest, index=weather_data.index, columns=['frest'])
        matrix_weather = create_matrix_weather(mode, weather_data=weather, restriction_data=rest_temp,
                                               rest_key='frest')
        matrix_weathers.append(matrix_weather)
        days_harvests.append(create_days_harvest(mode, matrix_weather, site))

    return params, doy_irr, matrix_weathers, days_harvests


def run_restrictions(storyline_key, outdir, nsims='all', mode_sites=default_mode_sites, padock_rest=False,
                     save_daily=False, description=''):
    storyline_path, swg_path, nsims_aval, simlen = storyline_swg_paths[storyline_key]

    if nsims == 'all':
        nsims = nsims_aval

    if nsims >= nsims_aval:
        print('not or just enough sims available, running all')
        nsims = nsims_aval
    else:
        pass

    number_run = (psutil.virtual_memory().available // memory_per_run)
    chunks = -1 * (-nsims // number_run)

    for mode, site in mode_sites:
        for c in range(chunks):
            _run_simple_rest(storyline_path, swg_path, nsims, mode, site, chunks, c, number_run, simlen, storyline_key,
                             outdir,
                             save_daily, description)

        if padock_rest:
            _run_paddock_rest(storyline_key, outdir, storyline_path, swg_path, nsims, mode, site, simlen,
                              save_daily, description)


def _run_simple_rest(storyline_path, swg_path, nsims, mode, site, chunks, c, number_run, simlen, storyline_key, outdir,
                     save_daily, description):
    params, doy_irr, all_matrix_weathers, all_days_harvests = _gen_input(storyline_path, swg_path,
                                                                         nsims=nsims, mode=mode, site=site,
                                                                         chunks=chunks, current_c=c,
                                                                         nperc=number_run, simlen=simlen)

    all_out = np.zeros((len(out_variables), simlen, number_run)) * np.nan
    for i, (matrix_weather, days_harvest) in enumerate(zip(all_matrix_weathers, all_days_harvests)):
        restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
        out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
        out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

        pg = pd.DataFrame(
            calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'PGR'] = pg.loc[:, 'pg']
        out.loc[:, 'F_REST'] = restrict

        all_out[:, :, i] = out.loc[:, out_variables].values.transpose()  # todo check

    # one netcdf file for each mode/site
    month = out.loc[:, 'month'].values
    doy = out.loc[:, 'doy'].values
    year = out.loc[:, 'year'].values
    _output_to_nc(storyline_key=storyline_key, outdir=outdir, outdata=all_out, nsims=nsims,
                  month=month, doy=doy, year=year,
                  chunks=chunks, current_c=c, nperchunk=number_run, save_daily=save_daily,
                  description=description)


def _run_paddock_rest(storyline_key, outdir, storyline_path, swg_path, nsims, mode, site, simlen,
                      save_daily, description):
    """
    run storyline through paddock restrictions...
    :param storyline_path:
    :param swg_path:
    :param nsims:
    :param mode:
    :param site:
    :return:
    """
    # paddock level restrictions
    levels = np.arange(0, 125, 25) / 100
    number_run = (psutil.virtual_memory().available // (memory_per_run * len(levels)))
    chunks = -1 * (-nsims // number_run)

    for c in range(chunks):
        params, doy_irr, all_matrix_weathers, all_days_harvests = _gen_input(storyline_path, swg_path,
                                                                             nsims=nsims, mode=mode, site=site,
                                                                             chunks=chunks, current_c=c,
                                                                             nperc=number_run, simlen=simlen)

        all_out = np.zeros((len(out_variables), simlen, len(levels) - 1, number_run)) * np.nan
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

                temp = run_basgra_nz(params, matrix_weather_new, days_harvest, doy_irr, verbose=False)
                temp.loc[:, 'PER_PAW'] = temp.loc[:, 'PAW'] / temp.loc[:, 'MXPAW']
                temp.loc[:, 'PGR'] = calc_pasture_growth(temp, days_harvest, 'from_yield', '1D', resamp_fun='mean')
                temp.loc[:, 'F_REST'] = restrict

                all_out[:, :, i, j] = temp.loc[:, out_variables].values.transpose()  # todo check
        # one netcdf for each mode,site, (paddock, mean)
        month = temp.loc[:, 'month'].values
        doy = temp.loc[:, 'doy'].values
        year = temp.loc[:, 'year'].values
        _output_to_nc_paddock(storyline_key=storyline_key, outdir=outdir, outdata=all_out, nsims=nsims,
                              month=month, doy=doy, year=year,
                              chunks=chunks, current_c=c, nperchunk=number_run,
                              out_names=out_names, out_limits=out_limits,
                              save_daily=save_daily, description=description)


def _output_to_nc(storyline_key, outdir, outdata, nsims, month, doy, year,
                  chunks, current_c, nperchunk, save_daily=False, description=''):
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

    storyline_path, swg_path, nsims_aval, simlen = storyline_swg_paths[storyline_key]

    with open(storyline_path, 'r') as f:
        storyline_text = f.readlines()

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, '{}.nc'.format(storyline_key))

    # get netcdf file instance
    if current_c == 0:
        nc_file = _create_nc_file(outpath, number_run=nsims, month=month, doy=doy, year=year,
                                  storyline_text=storyline_text,
                                  save_daily=False, description=description)
    else:
        nc_file = nc.Dataset(outpath, 'a')

    _write_data(nc_file, year, month, outdata, current_c, nperchunk, chunks, nsims, save_daily)


def _output_to_nc_paddock(storyline_key, outdir, outdata, nsims, month, doy, year,
                          chunks, current_c, nperchunk,
                          out_names, out_limits,
                          save_daily=False, description=''):
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

    storyline_path, swg_path, nsims_aval, simlen = storyline_swg_paths[storyline_key]

    with open(storyline_path, 'r') as f:
        storyline_text = f.readlines()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # outdata shape np.zeros((len(out_variables), simlen, len(levels) - 1, number_run))
    for i, (outn, outl) in enumerate(zip(out_names, out_limits)):
        outpath = os.path.join(outdir, '{}-{}.nc'.format(storyline_key, outn))
        extra = ('\npaddock restrictions are broken into {} different paddocks, where '.format(len(out_names)) +
                 'each paddock absorbs specific restrictions defined by limts:\n')
        description = description + extra + '\n' + outn + '\n' + outl

        # get netcdf file instance
        if current_c == 0:
            nc_file = _create_nc_file(outpath, number_run=nsims, month=month, doy=doy, year=year,
                                      storyline_text=storyline_text,
                                      save_daily=False, description=description)
        else:
            nc_file = nc.Dataset(outpath, 'a')

        _write_data(nc_file, year, month, outdata[:, :, i, :], current_c, nperchunk, chunks, nsims, save_daily)

    # save the mean data
    outpath = os.path.join(outdir, '{}-mean.nc'.format(storyline_key))
    extra = ('\npaddock restrictions are broken into {} different paddocks, where '.format(len(out_names)) +
             'each paddock absorbs specific restrictions defined by limts this file represents the mean of all'
             'paddocks')
    description = description + extra + '\n'

    # get netcdf file instance
    if current_c == 0:
        nc_file = _create_nc_file(outpath, number_run=nsims, month=month, doy=doy, year=year,
                                  storyline_text=storyline_text,
                                  save_daily=False, description=description)
    else:
        nc_file = nc.Dataset(outpath, 'a')

    outdata = np.nanmean(outdata, axis=2)
    _write_data(nc_file, year, month, outdata, current_c, nperchunk, chunks, nsims, save_daily)


def _write_data(nc_file, year, month, outdata, current_c, nperchunk, chunks, nsims, save_daily):
    # write monthly data chunk
    for i, v in enumerate(out_variables):
        temp = nc_file.variables[v]
        # outdata shape = np.zeros((len(out_variables), simlen_monthly, number_run))

        # resample to monthly data
        yearmonths = list(pd.Series(zip(year, month)).unique())
        resamp_data = np.zeros((outdata.shape[1], len(yearmonths))) * np.nan
        for k, (y, m) in enumerate(yearmonths):
            idx = ((year == y) & (month == m))
            resamp_data[k, :] = np.nanmean(outdata[i, idx, :], axis=0)

        to_val = (current_c + 1) * nperchunk
        # capture last chunk...
        if current_c + 1 == chunks:
            to_val = nsims
        temp[:, (current_c * nperchunk): to_val] = outdata[i, :, :]

        # write daily data chunk
    if save_daily:
        for i, v in enumerate(out_variables):
            temp = nc_file.variables[v]
            to_val = (current_c + 1) * nperchunk
            # capture last chunk...
            if current_c + 1 == chunks:
                to_val = nsims

            # outdata shape = np.zeros((len(out_variables), simlen, number_run))
            temp[:, (current_c * nperchunk): to_val] = outdata[i, :, :]


def _create_nc_file(outpath, number_run, month, doy, year, storyline_text, save_daily=False, description=''):
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
    for v in out_variables:
        temp = nc_file.createVariable('m_{}'.format(v), float, ('sim_month', 'realisation'))
        temp.setncatts({'units': out_metadata[v.upper()]['unit'],
                        'description': out_metadata[v.upper()]['description'],
                        'long_name': 'monthly mean {}'.format(v),
                        'missing_value': np.nan})

        if save_daily:
            temp2 = nc_file.createVariable('d_{}'.format(v), float, ('sim_month', 'realisation'))
            temp2.setncatts({'units': out_metadata[v.upper()]['unit'],
                             'description': out_metadata[v.upper()]['description'],
                             'long_name': 'daily raw {}'.format(v),
                             'missing_value': np.nan})

    return nc_file

# todo calculate pasture growth anamoly??? or external function?, external function but defined here
