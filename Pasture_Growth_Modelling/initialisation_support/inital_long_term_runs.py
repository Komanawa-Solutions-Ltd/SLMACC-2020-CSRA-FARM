"""
 Author: Matt Hanson
 Created: 25/11/2020 11:42 AM
 """

import pandas as pd
import numpy as np
import os
import ksl_env

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from basgra_python import run_basgra_nz
from supporting_functions.plotting import plot_multiple_results, plot_multiple_monthly_results
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth, calc_pasture_growth_anomaly
from Pasture_Growth_Modelling.initialisation_support.comparison_support import make_mean_comparison, \
    get_horarata_data_old, get_indicative_irrigated


def run_past_basgra_irrigated(return_inputs=False, site='eyrewell', reseed=True):
    mode = 'irrigated'
    print('running: {}, {}, {}'.format(mode, site, reseed))
    weather = get_vcsn_record(site)
    rest = get_restriction_record()
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather, site)

    if not reseed:
        days_harvest.loc[:, 'reseed_trig'] = -1

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

    pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
    out.loc[:, 'pg'] = pg.loc[:, 'pg']
    out = calc_pasture_growth_anomaly(out, fun='mean')

    if return_inputs:
        return out, (params, doy_irr, matrix_weather, days_harvest)
    return out


def run_past_basgra_dryland(return_inputs=False, site='eyrewell', reseed=True):
    mode = 'dryland'
    print('running: {}, {}, reseed: {}'.format(mode, site, reseed))
    weather = get_vcsn_record(site)
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest)
    days_harvest = create_days_harvest(mode, matrix_weather, site)
    if not reseed:
        days_harvest.loc[:, 'reseed_trig'] = -1

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
    out.loc[:, 'pg'] = pg.loc[:, 'pg']
    out = calc_pasture_growth_anomaly(out, fun='mean')

    if return_inputs:
        return out, (params, doy_irr, matrix_weather, days_harvest)
    return out


if __name__ == '__main__':
    outdir = ksl_env.shared_drives(r"SLMACC_2020\pasture_growth_modelling\historical_runs")
    save = True
    data = {
        'irrigated_eyrewell': run_past_basgra_irrigated(),
        'irrigated_oxford': run_past_basgra_irrigated(site='oxford'),
        'dryland_eyrewell': run_past_basgra_dryland(),
        'dryland_oxford': run_past_basgra_dryland(site='oxford'),
    }
    for i, k in enumerate(data.keys()):
        data[k].loc[:, 'RESEEDED'] += i #todo any more fo these to raise up?

    data2 = {e: make_mean_comparison(v, 'mean') for e, v in data.items()}
    data2['Horoata'] = get_horarata_data_old()
    data2['indicative_irr'] = get_indicative_irrigated()

    for k, v in data2.items():
        v.loc[:,'month'] = v.index.month
        v.set_index('month', inplace=True)

    out_vars = ['DM', 'DMH', 'YIELD', 'DMH_RYE', 'DM_RYE_RM', 'DMH_WEED', 'DM_WEED_RM', 'IRRIG', 'RAIN', 'EVAP', 'TRAN',
                'per_PAW', 'pg', 'RESEEDED',
                'pga_norm', 'BASAL']

    data3 = {e: v.groupby('month').mean() for e, v in data.items()}
    plt_outdir_sim = None
    plt_outdir_aver_yr = None
    if save:
        plt_outdir_sim = os.path.join(outdir, 'plots', 'full_hist')
        plt_outdir_aver_yr = os.path.join(outdir, 'plots', 'aver_yr')
        for d in [plt_outdir_aver_yr, plt_outdir_sim]:
            if not os.path.exists(d):
                os.makedirs(d)
        for k, v in data.items():
            v.to_csv(os.path.join(outdir, '{}_raw.csv'.format(k)))
            v.resample('10D').mean().to_csv(os.path.join(outdir, '{}_10daily.csv'.format(k)))
            v.resample('M').mean().to_csv(os.path.join(outdir, '{}_monthly.csv'.format(k)))
            v.groupby('month').mean().to_csv(os.path.join(outdir, '{}_average_year.csv'.format(k)))

    # shift to jan in the middle for average year

    plot_multiple_results(data=data, out_vars=out_vars, rolling=90, label_rolling=True, label_main=False,
                          main_kwargs={'alpha': 0.2},
                          show=False, outdir=plt_outdir_sim, title_str='historical_')
    plot_multiple_monthly_results(data=data3, out_vars=out_vars, show=False, outdir=plt_outdir_aver_yr,
                          title_str='average_year_',  main_kwargs={'marker':'o'})
    plot_multiple_monthly_results(data=data2, out_vars=['pg'], show=(not save), outdir=plt_outdir_aver_yr,
                          title_str='cumulative_average_year_', main_kwargs={'marker':'o'})
