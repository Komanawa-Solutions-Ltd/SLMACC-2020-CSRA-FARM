"""
 Author: Matt Hanson
 Created: 26/02/2021 9:54 AM
 """
import os
import numpy as np
import pandas as pd
import time
import datetime
from copy import deepcopy
import multiprocessing
import logging
import psutil
from Ecological_flows.river_flow_pgr_model import run_pasture_growth_river_flow, default_swg_dir, default_mode_sites, \
    default_riv_pasture_growth_dir, make_current_restrictions

pgm_log_dir = os.path.join(default_riv_pasture_growth_dir, 'logs')
if not os.path.exists(pgm_log_dir):
    os.makedirs(pgm_log_dir)


def run_full_model_mp(storyline_path_mult,
                      outdir_mult,
                      nsims_mult,
                      log_path,
                      description_mult,
                      rest_funs_mult,
                      save_daily_mult=False,
                      mode_sites_mult=default_mode_sites,
                      swg_dir_mult=default_swg_dir,
                      pool_size=None,
                      fix_leap=True,
                      verbose=False,
                      re_run=True,
                      seed=None,
                      use_1_seed=False):
    """
    run a bunch of basgra models for storylines
    :param storyline_path_mult: list of storyline_paths
    :param outdir_mult: list of outdirs
    :param nsims_mult: can be either a single value or a list
    :param mode_sites_mult: can be either a single value or a list
    :param rest_funs_mult: function or list of functions to calc restriction
    :param save_daily_mult: can be either a single value or a list
    :param description_mult: can be either a single value or a list
    :param swg_dir_mult: can be either a single value or a list
    :param log_path: path to save the log, has the time and .csv appended to it
    :param pool_size: if none use full processor pool otherwise specify
    :param seed: random seeds for each storyline
    :param use_1_seed: bool if True use the same seed for each month of the storyline, if false calculate a new seed
                       for each month of each storyline.  The new seeds are based on the passed seed so they are
                       reproducable (via seed kwarg); however each month will have different data.
    :return:
    """
    if verbose:
        sp = start_process
    else:
        sp = silent_start_process
    log_path = f'{log_path}-{datetime.datetime.now().isoformat().replace(":", "-").split(".")[0]}.csv'
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    storyline_path_mult = np.atleast_1d(storyline_path_mult)
    outdir_mult = np.atleast_1d(outdir_mult)
    ex_shape = storyline_path_mult.shape
    assert storyline_path_mult.ndim == 1
    assert ex_shape == outdir_mult.shape

    # make args that might not be the right shape into the right shape
    args_n1d = {
        'nsims_mult': nsims_mult,
        'mode_sites_mult': mode_sites_mult,
        'rest_funs_mult': rest_funs_mult,
        'save_daily_mult': save_daily_mult,
        'description_mult': description_mult,
        'swg_dir_mult': swg_dir_mult,
        'use_1_seed': use_1_seed,
        'seed': seed,

    }
    for k in args_n1d.keys():
        if k == 'mode_sites_mult':
            t = np.array(args_n1d[k])
            if t.ndim == 2:
                t = np.moveaxis(np.atleast_3d(t), -1, 0)
                t = np.array([t[0] for e in range(ex_shape[0])])
            elif t.ndim == 3 and len(t) == 1:
                t = np.array([t[0] for e in range(ex_shape[0])])

            assert t.shape[0] == ex_shape[0] and t.shape[-1] == 2, (f'unexpected shape{t.shape} for {k} '
                                                                    'expected to match storylines '
                                                                    f'shape ({ex_shape[0]},2) '
                                                                    'or to be of length 1 or non-iterable')

        else:
            t = np.atleast_1d(args_n1d[k])
            if len(t) == 1:
                t = np.repeat(t, ex_shape)
            assert t.shape == ex_shape, (f'unexpected shape{t.shape} for {k} '
                                         f'expected to match storylines shape {ex_shape} '
                                         'or to be of length 1 or non-iterable')
        args_n1d[k] = t

    if pool_size is None:
        pool_size = psutil.cpu_count(logical=True)
    pool_size = min(ex_shape[0],
                    pool_size)  # to make is so full memory is avalible if running a smaller num of sims than pool size

    runs = []
    for i, (s, o) in enumerate(zip(storyline_path_mult, outdir_mult)):
        runs.append(deepcopy({
            'storyline_path': s,
            'outdir': o,
            'nsims': args_n1d['nsims_mult'][i],
            'mode_sites': args_n1d['mode_sites_mult'][i],
            'rest_fun': args_n1d['rest_funs_mult'][i],
            'save_daily': args_n1d['save_daily_mult'][i],
            'description': args_n1d['description_mult'][i],
            'swg_dir': args_n1d['swg_dir_mult'][i],
            'verbose': verbose,
            'n_parallel': pool_size,
            'fix_leap': fix_leap,
            're_run': re_run,
            'seed': args_n1d['seed'][i],
            'use_1_seed': args_n1d['use_1_seed'][i],

        }))
    t = time.time()
    multiprocessing.log_to_stderr(logging.DEBUG)
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=sp)

    results = pool.map_async(_rpg_mp, runs)
    pool_outputs = results.get()
    pool.close()  # no more tasks
    pool.join()
    print('completed {} runs in {} min'.format(len(runs), (time.time() - t) / 60))

    outdata = pd.DataFrame(pool_outputs,
                           columns=kwarg_keys)
    outdata.to_csv(log_path)


def start_process():
    """
    function to run at the start of each multiprocess sets the priority lower
    :return:
    """
    print('Starting', multiprocessing.current_process().name)
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def silent_start_process():
    """
    function to run at the start of each multiprocess sets the priority lower
    :return:
    """
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


kwarg_keys = ['storyline_path', 'success', 'outdir', 'nsims', 'mode_sites', 'padock_rest', 'save_daily', 'description',
              'swg_dir', 'error_v']


def _rpg_mp(kwargs):
    storyline_id = os.path.basename(kwargs['storyline_path']).split('.')[0]
    kwargs['nsims'] = int(kwargs['nsims'])

    success = True
    v = 'no error'

    print('starting to run {}'.format(storyline_id))
    try:
        run_pasture_growth_river_flow(**kwargs)
    except Exception as val:
        v = val
        success = False
    kwargs['success'] = success
    kwargs['error_v'] = v
    kwargs['mode_sites'] = '; '.join(['-'.join(e) for e in kwargs['mode_sites']])
    print('finished storyline: {},  success: {}, error: {}'.format(storyline_id, success, v))
    return [kwargs[e] for e in kwarg_keys]


if __name__ == '__main__':
    from Climate_Shocks.climate_shocks_env import storyline_dir

    # todo check

    nsims = 1
    ex_run_fun = make_current_restrictions
    run_mp = True
    run_norm = False

    spaths = np.repeat([os.path.join(storyline_dir, '0-baseline.csv')], (10))
    odirs = [os.path.join(default_riv_pasture_growth_dir, 'test_mp_f', f't{e}') for e in range(10)]

    if run_mp:
        run_full_model_mp(storyline_path_mult=spaths,
                          outdir_mult=odirs,
                          nsims_mult=nsims,
                          log_path=os.path.join(pgm_log_dir, 'test_mp_function.csv'),
                          description_mult='just to test mp function',
                          save_daily_mult=False,
                          pool_size=None,
                          verbose=True,
                          rest_funs_mult=ex_run_fun)
    if run_norm:
        for p, od in zip(spaths, odirs):
            print(p)
            run_pasture_growth_river_flow(storyline_path=p, outdir=od, nsims=nsims, rest_fun=ex_run_fun,
                                          save_daily=True, description='', verbose=True,
                                          n_parallel=1)
