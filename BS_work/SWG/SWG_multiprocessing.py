"""
 Author: Matt Hanson
 Created: 22/02/2021 10:28 AM
 """
import os
import numpy as np
import pandas as pd
import time
import multiprocessing
import logging
import psutil
from BS_work.SWG.SWG_wrapper import create_yaml, run_SWG, oxford_lon, oxford_lat


def run_swg_mp(storyline_paths, outdirs, ns, base_dirs, vcfs, cleans, log_path):  # todo check!
    """
    run a bunch of storylines as multiprocessing
    :param storyline_paths: storyline paths to run
    :param outdirs: out directories to run
    :param ns: number of sims, can be just int
    :param base_dirs: base_dirs can be just single path
    :param vcfs: vcfs can be single path
    :param cleans: can be 1 boolean
    :param log_path: directory to save the log
    :return:
    """
    storyline_paths = np.atleast_1d(storyline_paths)
    outdirs = np.atleast_1d(outdirs)
    assert storyline_paths.ndim == 1
    assert storyline_paths.shape == outdirs.shape

    ns = np.atleast_1d(ns)
    if len(ns) == 1:
        ns = np.repeat(ns, storyline_paths.shape).astype(int)

    base_dirs = np.atleast_1d(base_dirs)
    if len(base_dirs) == 1:
        base_dirs = np.repeat(base_dirs, storyline_paths.shape)

    vcfs = np.atleast_1d(vcfs)
    if len(vcfs) == 1:
        vcfs = np.repeat(vcfs, storyline_paths.shape)

    cleans = np.atleast_1d(cleans)
    if len(cleans) == 1:
        cleans = np.repeat(cleans, storyline_paths.shape)

    assert ns.shape == base_dirs.shape == vcfs.shapes == cleans.shape == storyline_paths.shape

    runs = []
    # make into dictionary
    for s, o, n, b, v, c in zip(storyline_paths, outdirs, ns, base_dirs, vcfs, cleans):
        runs.append({
            'storyline_path': s,
            'outdir': o,
            'n': n,
            'base_dir': b,
            'vcf': v,
            'clean': c,
        })
    t = time.time()
    multiprocessing.log_to_stderr(logging.DEBUG)
    pool_size = psutil.cpu_count(logical=True)
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

    results = pool.map_async(_run_mp, runs)
    while not results.ready():
        time.sleep(60)  # sleep 1 min between printing
    pool_outputs = results.get()
    pool.close()  # no more tasks
    pool.join()
    print('completed {} runs in {} min'.format(len(runs), (time.time() - t) / 60))

    outdata = pd.DataFrame(pool_outputs,
                           columns=['storyline_path', 'outdir', 'n', 'base_dir', 'vcf', 'clean', 'success', 'errors'])
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
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


def _run_mp(**kwargs):
    # set up args
    storyline_path = kwargs['storyline_path']
    outdir = kwargs['outdir']
    n = kwargs['n']
    base_dir = kwargs['base_dir']
    vcf = kwargs['vcf']
    clean = kwargs['clean']

    yml = os.path.join(outdir, 'ind.yml')
    success = True
    v = ''
    try:
        create_yaml(outpath_yml=yml, outsim_dir=outdir,
                    nsims=n,
                    storyline_path=storyline_path,
                    sim_name=None,
                    xlat=oxford_lat, xlon=oxford_lon,
                    base_dir=base_dir,
                    vcf=vcf)
        temp = run_SWG(yml, outdir, rm_npz=True, clean=clean)
    except Exception as v:
        success = False
    return storyline_path, outdir, n, base_dir, vcf, clean, success, v
