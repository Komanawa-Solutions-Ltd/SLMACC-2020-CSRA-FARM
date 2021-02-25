"""
 Author: Matt Hanson
 Created: 26/02/2021 9:54 AM
 """
import os
import numpy as np
import pandas as pd
import time
import multiprocessing
import logging
import psutil
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth, default_swg_dir, default_mode_sites

# todo check full script
def run_full_model_mp(storyline_path_mult,
                      outdir_mult,
                      nsims_mult,
                      mode_sites_mult,
                      padock_rest_mult,
                      save_daily_mult,
                      description_mult,
                      swg_dir_mult,
                      log_path, pool_size=None):
    """
    run a bunch of BASGRA models for storylines as multiprocessing
    :param log_path: directory to save the log
    :param pool_size: if none use full processor pool otherwise specify
    :return:
    """
    storyline_path_mult = np.atleast_1d(storyline_path_mult)
    outdir_mult = np.atleast_1d(outdir_mult)
    ex_shape = storyline_path_mult.shape
    assert storyline_path_mult.ndim == 1
    assert ex_shape == outdir_mult.shape

    # might not be 1d
    args_n1d = {
        'nsims_mult': nsims_mult,
        'mode_sites_mult': mode_sites_mult,
        'padock_rest_mult': padock_rest_mult,
        'save_daily_mult': save_daily_mult,
        'description_mult': description_mult,
        'swg_dir_mult': swg_dir_mult,

    }
    for k in args_n1d.keys():
        t = np.atleast_1d(args_n1d[k])
        if len(t) == 1:
            t = np.repeat(t, ex_shape)
        assert t.shape == ex_shape, (f'unexpected shape for {k} '
                                                  f'expected to match storylines shape {ex_shape} '
                                                  'or to be of length 1 or non-iterable')
        args_n1d[k] = t

    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    runs = []
    for i, (s, o) in enumerate(zip(storyline_path_mult, outdir_mult)):
        runs.append({
            'storyline_path': s,
            'outdir': o,
            'nsims': args_n1d['nsims_mult'][i],
            'mode_sites': args_n1d['mode_sites_mult'][i],
            'padock_rest': args_n1d['padock_rest_mult'][i],
            'save_daily': args_n1d['save_daily_mult'][i],
            'description': args_n1d['description_mult'][i],
            'swg_dir': args_n1d['swg_dir_mult'][i],
        })
    t = time.time() #todo check args are right
    multiprocessing.log_to_stderr(logging.DEBUG)
    if pool_size is None:
        pool_size = psutil.cpu_count(logical=True)
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process)

    results = pool.map_async(_rpg_mp, runs)
    while not results.ready():
        time.sleep(60)  # sleep 1 min between printing
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


kwarg_keys = ['storyline_path', 'success', 'outdir', 'nsims', 'mode_sites', 'padock_rest', 'save_daily', 'description',
              'swg_dir', 'error_v']


def _rpg_mp(kwargs):
    storyline_id = os.path.basename(kwargs['storyline_path']).split('.')[0]

    success = True
    v = 'no error'

    print('starting to run {}'.format(storyline_id))
    try:
        run_pasture_growth(**kwargs)
    except Exception as val:
        v = val
        success = False
    kwargs['success'] = success
    kwargs['error_v'] = v
    print('finished storyline: {},  success: {}, error: {}'.format(storyline_id, success, v))
    return [kwargs[e] for e in kwarg_keys]
