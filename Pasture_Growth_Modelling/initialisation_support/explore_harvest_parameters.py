"""
 Author: Matt Hanson
 Created: 23/11/2020 11:02 AM
 """

import project_base
import pandas as pd
import numpy as np
import os
from komanawa.basgra_nz_py.example_data import get_lincoln_broadfield, get_woodward_weather, clean_harvest, establish_org_input
from komanawa.basgra_nz_py.basgra_python import run_basgra_nz
from komanawa.basgra_nz_py.supporting_functions.plotting import plot_multiple_results


def run_old_basgra():
    params, matrix_weather, days_harvest, doy_irr = establish_org_input('lincoln')

    days_harvest = clean_harvest(days_harvest, matrix_weather)

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    return out


def run_frequent_harvest(freq, trig, targ):
    params, matrix_weather, days_harvest, doy_irr = establish_org_input('lincoln')

    strs = ['{}-{:03d}'.format(e, f) for e, f in matrix_weather[['year', 'doy']].itertuples(False, None)]
    days_harvest = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                 'doy': matrix_weather.loc[:, 'doy'],
                                 'frac_harv': np.ones(len(matrix_weather)),  # set filler values
                                 'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                 'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                 'weed_dm_frac': np.zeros(len(matrix_weather)),  # set filler values
                                 })

    # start harvesting at the same point
    harv_days = pd.date_range(start='2011-09-03', end='2017-04-30', freq='{}D'.format(freq))
    idx = np.in1d(pd.to_datetime(strs, format='%Y-%j'), harv_days)
    days_harvest.loc[idx, 'harv_trig'] = trig
    days_harvest.loc[idx, 'harv_targ'] = targ

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    return out


if __name__ == '__main__':
    outdir = ksl_env.slmmac_dir.joinpath(r"pasture_growth_modelling/basgra_harvest_tuning/irr_harv_testing")
    data = {
        'Woodward_model': run_old_basgra(),
    }
    freq = [10, 10, 10]
    trigs = [1501, 1600, 1700]
    targs = [1500, 1500, 1500]

    for f, tr, ta, in zip(freq, trigs, targs):
        data['freq: {}, Trig:{}, Targ:{}'.format(f, tr, ta)] = run_frequent_harvest(f, tr, ta)

    plot_multiple_results(data, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM'], outdir=os.path.join(outdir, 'trig_vary'))

    data = {
        'Woodward_model': run_old_basgra(),
    }
    freq = [10, 20, 30]
    trigs = [1501, 1501, 1501]
    targs = [1500, 1500, 1500]

    for f, tr, ta, in zip(freq, trigs, targs):
        data['freq: {}, Trig:{}, Targ:{}'.format(f, tr, ta)] = run_frequent_harvest(f, tr, ta)

    plot_multiple_results(data, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM'],outdir=os.path.join(outdir, 'freq_vary'))

    data = {
        'Woodward_model': run_old_basgra(),
    }
    freq = [10, 10, 10]
    trigs = [1301, 1501, 1801]
    targs = [1300, 1500, 1800]

    for f, tr, ta, in zip(freq, trigs, targs):
        data['freq: {}, Trig:{}, Targ:{}'.format(f, tr, ta)] = run_frequent_harvest(f, tr, ta)

    plot_multiple_results(data, out_vars=['DM', 'YIELD', 'BASAL', 'DMH_RYE', 'DM_RYE_RM'], outdir=os.path.join(outdir, 'trag_vary'))
