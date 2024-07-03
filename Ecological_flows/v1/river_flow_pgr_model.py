"""
 Author: Matt Hanson
 Created: 12/10/2021 9:24 AM
 """
import pandas as pd
import project_base
from Climate_Shocks.Stochastic_Weather_Generator.irrigation_flow_generator import get_irrigation_generator, \
    make_current_restrictions
from Pasture_Growth_Modelling.full_model_implementation import default_mode_sites, default_swg_dir, month_len, \
    memory_per_run, abs_max_irr, run_basgra_nz, calc_pasture_growth, _output_to_nc, get_params_doy_irr, \
    _get_weather_data, create_matrix_weather, create_days_harvest, get_rest_tolerance
import numpy as np
import os
import time
import psutil

out_variables_flow = (
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
    'FLOW',
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

irr_flow_gen = get_irrigation_generator()

default_riv_pasture_growth_dir = os.path.join(os.path.join(ksl_env.unbacked_dir, 'river_pgr_sims'))
os.makedirs(default_riv_pasture_growth_dir, exist_ok=True)


def get_river_flow_from_storyline(num_to_pull, storyline, rest_fun, simlen, seed=None, use_1_seed=False):
    """

    :param num_to_pull: number of realisataions to pull
    :param storyline: storyline data (pd. Dataframe)
    :param simlen: length of the simulation
    :param rest_fun: function to change flow to restriction record
    :param seed: seed for random functions
    :param use_1_seed: all of the seeds for different months will be identical. ,
                       useful for comparison of linked - non linked
    :return:
    """
    if use_1_seed:
        seeds = np.repeat(seed, len(storyline) * 10)  # to allow linked comparison
    else:
        np.random.seed(seed)
        seeds = np.random.randint(0, 3548324, len(storyline) * 10)  # make too many
    flow = np.zeros((num_to_pull, simlen))
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
        flow[:, idx: idx + month_len[m]] = irr_flow_gen.get_data(num_to_pull, key=key,
                                                                 suffix='rest',
                                                                 suffix_selection=r,
                                                                 tolerance=get_rest_tolerance(r),
                                                                 max_replacement_level=0.1,
                                                                 under_level='warn',
                                                                 seed=seeds[i])
        idx += month_len[m]

    rest = rest_fun(flow)
    # tod I need to add the naturalisation into this (and or the increased takes)...
    return rest, flow


def get_rest_river_output_from_storyline_path(storyline_path, num_to_pull, rest_fun, seed=None, use_1_seed=False,
                                              fix_leap=True):  # tod check
    storyline = pd.read_csv(storyline_path)
    simlen = np.array([month_len[e] for e in storyline.month]).sum()
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

    rest, flow = get_river_flow_from_storyline(num_to_pull, storyline, rest_fun, simlen, seed=seed,
                                               use_1_seed=use_1_seed)
    # rest,flow is (stamples, simlen)
    outdata = pd.DataFrame(index=out_index, data=np.concatenate([rest, flow], axis=0).transpose(),
                           columns=np.concatenate((
                               [f'rest_{e:03d}' for e in range(num_to_pull)],
                               [f'flow_{e:03d}' for e in range(num_to_pull)],
                           )))
    outcols = []
    for e in range(num_to_pull):
        outcols.extend([f'flow_{e:03d}', f'rest_{e:03d}'])
    outdata = outdata.loc[:, outcols]
    return outdata


def run_pasture_growth_river_flow(storyline_path, outdir, nsims, rest_fun, mode_sites=default_mode_sites,
                                  save_daily=True, description='', swg_dir=default_swg_dir, verbose=True,
                                  n_parallel=1, fix_leap=True, re_run=True, seed=None, use_1_seed=False,
                                  use_out_variables=out_variables_flow):  # tod check
    """
    creates weather data, runs basgra and saves values to a netcdf
    :param storyline_path: path to the storyline
    :param outdir: directory to save the files to
    :param nsims: number of simulations to run
    :param rest_fun: function to change flow to restriction record
    :param mode_sites: a list of modes and sites some or all of:     ('dryland', 'oxford'),
                                                                     ('irrigated', 'eyrewell'),
                                                                     ('irrigated', 'oxford'),
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
    storyline_key = os.path.splitext(os.path.basename(storyline_path))[0]
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
                         rest_fun=rest_fun,
                         use_out_variables=use_out_variables)
    t = time.time() - t
    if verbose:
        print(f'took {t / 60} min to run {nsims} sims')


def _run_simple_rest(storyline, nsims, mode, site, simlen, storyline_key, outdir,
                     save_daily, description, storyline_text, swg_dir, verbose, n_parallel, fix_leap, seed,
                     use_1_seed, rest_fun,
                     use_out_variables):
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
        params, doy_irr, all_matrix_weathers, all_days_harvests, flow, rest_data = _gen_input(
            storyline=storyline, nsims=nsims, mode=mode, site=site, chunks=chunks, current_c=c,
            nperc=number_run, simlen=simlen, swg_dir=swg_dir, fix_leap=fix_leap,
            seed=seed, use_1_seed=use_1_seed, rest_fun=rest_fun)

        all_out = np.zeros((len(use_out_variables), simlen, number_run)) * np.nan
        for i, (matrix_weather, days_harvest) in enumerate(zip(all_matrix_weathers, all_days_harvests)):
            restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
            assert np.isclose(restrict, rest_data[i]).all()  # tod internal check see if this causes problems!
            out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False, run_365_calendar=fix_leap)
            out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

            pg = pd.DataFrame(
                calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
            out.loc[:, 'PGR'] = pg.loc[:, 'pg']
            out.loc[:, 'F_REST'] = restrict
            out.loc[:, 'FLOW'] = flow[i]

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
                                use_out_variables=use_out_variables)  # flow and rest are both saved to the data already


def _gen_input(storyline, nsims, mode, site, chunks, current_c, nperc, simlen, swg_dir, fix_leap,
               rest_fun, seed=None, use_1_seed=False):
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
        rest_data, flow = np.repeat([None], num_to_pull), np.repeat([None], num_to_pull)
    elif mode == 'irrigated':
        rest_data, flow = get_river_flow_from_storyline(num_to_pull=num_to_pull, storyline=storyline, rest_fun=rest_fun,
                                                        simlen=simlen, seed=seeds[iseed], use_1_seed=use_1_seed)
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

    return params, doy_irr, matrix_weathers, days_harvests, flow, rest_data
