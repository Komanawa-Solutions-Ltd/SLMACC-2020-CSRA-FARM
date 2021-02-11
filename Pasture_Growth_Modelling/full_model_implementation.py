"""
 Author: Matt Hanson
 Created: 18/12/2020 11:23 AM
 """

import pandas as pd
import numpy as np
import os
import ksl_env
import psutil
from Climate_Shocks.story_swg_iid_managment import storyline_swg_paths

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from basgra_python import run_basgra_nz
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather, default_mode_sites, abs_max_irr
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth

# todo check that the SWG realisations fit our criterion!?!
# todo consider multiprocessing here????
# todo calculate pasture growth anamoly??? or external function?

memory_per_run = 1  # todo define in bytes

out_variables = (
    'BASAL',  # todo fill out
    'PGR',
)

def _gen_input(storyline_path, SWG_path, nsims, mode, site, chunks, current_c, nperc):
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
    #todo manage chunks
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weathers = []
    days_harvests = []

    # todo get restriction data and weather data
    rest_data = None
    weather_data = None

    # make all the other data
    for rest, weather in zip(rest_data, weather_data):
        matrix_weather = create_matrix_weather(mode, weather_data=weather, restriction_data=rest, rest_key=None)
        matrix_weathers.append(matrix_weather)
        days_harvests.append(create_days_harvest(mode, matrix_weather, site))

    return params, doy_irr, matrix_weathers, days_harvests


def run_restrictions(storyline_key, outdir, nsims=all, mode_sites=default_mode_sites, padock_rest=False):
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
            params, doy_irr, all_matrix_weathers, all_days_harvests = _gen_input(storyline_path, swg_path,
                                                                                 nsims=nsims, mode=mode, site=site,
                                                                                 chunks=chunks, current_c=c,
                                                                                 nperc=number_run)

            all_out = np.zeros((len(out_variables), simlen, number_run)) * np.nan
            for i, (matrix_weather, days_harvest) in enumerate(zip(all_matrix_weathers, all_days_harvests)):
                restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
                out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
                out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

                pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
                out.loc[:, 'PGR'] = pg.loc[:, 'pg']
                out.loc[:, 'F_REST'] = restrict

                all_out[:, :, i] = out.loc[:, out_variables].values.transpose()  # todo check

            # one netcdf file for each mode/site
            _output_to_nc(storyline_key=storyline_key, outdir=outdir, outdata=all_out, chunks=chunks, current_c=c)

        if padock_rest:
            _run_paddock_rest(storyline_key, outdir, storyline_path, swg_path, nsims, mode, site, simlen)


def _run_paddock_rest(storyline_key, outdir, storyline_path, swg_path, nsims, mode, site, simlen):
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
    number_run = (psutil.virtual_memory().available // (memory_per_run*len(levels)))
    chunks = -1 * (-nsims // number_run)

    for c in range(chunks):
        params, doy_irr, all_matrix_weathers, all_days_harvests = _gen_input(storyline_path, swg_path,
                                                                             nsims=nsims, mode=mode, site=site,
                                                                             chunks=chunks, current_c=c,
                                                                             nperc=number_run)

        all_out = np.zeros((len(out_variables), simlen, len(levels)-1, number_run)) * np.nan
        for j, (matrix_weather, days_harvest) in enumerate(zip(all_matrix_weathers, all_days_harvests)):
            restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
            matrix_weather.loc[:, 'max_irr'] = abs_max_irr
            for i, (ll, lu) in enumerate(zip(levels[0:-1], levels[1:])):
                out_name = 'paddock_{}'.format(i)
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
        _output_to_nc_paddock(storyline_key=storyline_key, outdir=outdir, outdata=all_out,  chunks=chunks, current_c=c)


def _output_to_nc(storyline_key, outdir, outdata, chunks, current_c):
    #todo
    raise NotImplementedError


def _output_to_nc_paddock(storyline_key, outdir, outdata, chunks, current_c):
    #todo
    raise NotImplementedError
