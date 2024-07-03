"""
 Author: Matt Hanson
 Created: 10/12/2020 9:15 AM
 """

import pandas as pd
import numpy as np
import os
import project_base
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# add basgra nz functions
from komanawa.basgra_nz_py.basgra_python import run_basgra_nz
from komanawa.basgra_nz_py.supporting_functions.plotting import plot_multiple_results, plot_multiple_monthly_results
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth, calc_pasture_growth_anomaly
from Pasture_Growth_Modelling.initialisation_support.validate_dryland_v2 import get_horarata_data_old, \
    make_mean_comparison
from Pasture_Growth_Modelling.initialisation_support.comparison_support import get_witchmore_mean, \
    make_mean_comparison_suite, make_total_suite, get_winchmore_boxplot
from Pasture_Growth_Modelling.initialisation_support.inital_long_term_runs import run_past_basgra_irrigated
from Pasture_Growth_Modelling.historical_average_baseline import run_past_basgra_dryland, run_past_basgra_irrigated


def run_mod_past_basgra_dryland(return_inputs=False, site='oxford', reseed=True, pg_mode='from_dmh', fun='mean',
                                reseed_trig=0.06, reseed_basal=0.1, basali=0.2, weed_dm_frac=0.05,
                                use_defined_params_except_weed_dm_frac=True):
    if not isinstance(weed_dm_frac, dict) and weed_dm_frac is not None:
        weed_dm_frac = {e: weed_dm_frac for e in range(1, 13)}
    mode = 'dryland'
    print('running: {}, {}, reseed: {}'.format(mode, site, reseed))
    weather = get_vcsn_record(site=site)
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest, fix_leap=False)
    days_harvest = create_days_harvest(mode, matrix_weather, site, fix_leap=False)
    if not reseed:
        days_harvest.loc[:, 'reseed_trig'] = -1
    else:
        if not use_defined_params_except_weed_dm_frac:
            days_harvest.loc[days_harvest.reseed_trig > 0, 'reseed_trig'] = reseed_trig
            days_harvest.loc[days_harvest.reseed_trig > 0, 'reseed_basal'] = reseed_basal
            params['BASALI'] = basali
    if weed_dm_frac is not None:
        for m in range(1, 13):
            days_harvest.loc[days_harvest.index.month == m, 'weed_dm_frac'] = weed_dm_frac[m]
    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode=pg_mode, resamp_fun=fun, freq='1d'))
    out.loc[:, 'pg'] = pg.loc[:, 'pg']
    out = calc_pasture_growth_anomaly(out, fun=fun)
    if return_inputs:
        return out, (params, doy_irr, matrix_weather, days_harvest)
    return out


def validation():
    fun = 'mean'

    weed_dict_1 = {
        1: 0.42,
        2: 0.30,
        3: 0.33,
        4: 0.32,
        5: 0.46,
        6: 0.59,
        7: 0.70,
        8: 0.57,
        9: 0.33,
        10: 0.37,
        11: 0.62,
        12: 0.62,
    }
    weed_dict_2 = {  # this is currently the calibration dataset
        1: 0.42,
        2: 0.30,
        3: 0.27,
        4: 0.25,
        5: 0.27,
        6: 0.20,
        7: 0.20,
        8: 0.20,
        9: 0.23,
        10: 0.30,
        11: 0.60,
        12: 0.62,
    }
    data = {
        'dryland_trended': run_past_basgra_dryland(site='oxford'),

        'weed: special2': run_mod_past_basgra_dryland(return_inputs=False, site='oxford', reseed=True,
                                                      pg_mode='from_yield',
                                                      fun='mean', reseed_trig=0.06, reseed_basal=0.1, basali=0.15,
                                                      weed_dm_frac=weed_dict_2,
                                                      use_defined_params_except_weed_dm_frac=True),

        'irrigated_oxford_trended': run_past_basgra_irrigated(site='oxford'),
        'irrigated_eyrewell_trended': run_past_basgra_irrigated(site='eyrewell'),

    }

    data2 = {e: make_mean_comparison(v, fun) for e, v in data.items()}
    data3 = {e: make_mean_comparison_suite(v, fun) for e, v in data.items()}
    data4 = {e: make_total_suite(v) for e, v in data.items()}
    data2['horata'] = get_horarata_data_old()
    data2['witchmore'] = get_witchmore_mean()
    out_vars = ['DM', 'YIELD', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'RAIN', 'EVAP', 'TRAN', 'per_PAW', 'pg', 'RESEEDED',
                'pga_norm', 'BASAL']
    if False:
        plot_multiple_results(data=data, out_vars=out_vars, rolling=90, label_rolling=True, label_main=False,
                              main_kwargs={'alpha': 0.8},
                              show=False)
    for k, v in data2.items():
        print(k, ': ', v.pg.sum())
    months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    for k, v in data3.items():
        fig, ax = plt.subplots()
        ax.boxplot(v['pgr'], labels=months)
        ax.set_ylim(-10, 110)
        ax.set_title(k)
    fig, ax = plt.subplots()
    temp_ks = data4.keys()
    temp_data = [data4[k].pg.values for k in temp_ks]
    ax.boxplot(temp_data, labels=temp_ks)

    plot_multiple_monthly_results(data=data2, out_vars=['pgr'], show=True, main_kwargs={'marker': 'o'})


def final_plots():
    fig_size = (10, 8)
    fun = 'mean'
    from pathlib import Path
    outdir = Path(project_base.slmmac_dir).joinpath("0_Y2_and_Final_Reporting/final_plots/dryland_validation")
    outdir.mkdir(exist_ok=True, parents=True)

    weed_dict_1 = {
        1: 0.42,
        2: 0.30,
        3: 0.33,
        4: 0.32,
        5: 0.46,
        6: 0.59,
        7: 0.70,
        8: 0.57,
        9: 0.33,
        10: 0.37,
        11: 0.62,
        12: 0.62,
    }
    weed_dict_2 = {  # this is currently the calibration dataset
        1: 0.42,
        2: 0.30,
        3: 0.27,
        4: 0.25,
        5: 0.27,
        6: 0.20,
        7: 0.20,
        8: 0.20,
        9: 0.23,
        10: 0.30,
        11: 0.60,
        12: 0.62,
    }
    data = {
        'Dryland Oxford trended': run_past_basgra_dryland(site='oxford'),

        'Irrigated Oxford trended': run_past_basgra_irrigated(site='oxford'),
        'Irrigated Eyrewell trended': run_past_basgra_irrigated(site='eyrewell'),

    }

    data2 = {e: make_mean_comparison(v, fun) for e, v in data.items()}
    data3 = {e: make_mean_comparison_suite(v, fun) for e, v in data.items()}
    data4 = {e: make_total_suite(v) for e, v in data.items()}
    data2['Hororata'] = get_horarata_data_old()
    data2['Winchmore'] = get_witchmore_mean()
    out_vars = ['DM', 'YIELD', 'DMH_RYE', 'DM_RYE_RM', 'IRRIG', 'RAIN', 'EVAP', 'TRAN', 'per_PAW', 'pg', 'RESEEDED',
                'pga_norm', 'BASAL']

    # plot uncertainty together for dryland and witchmore
    bplots = []
    colors = ['b', 'r']
    months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    fig, ax = plt.subplots(figsize=fig_size)
    x1 = list(range(0, 24, 2))
    bplots.append(ax.boxplot(data3['Dryland Oxford trended']['pgr'], positions=x1, patch_artist=True))

    # add winchmore to the boxplot
    wichmore = get_winchmore_boxplot()
    x2 = list(range(1, 24, 2))
    bplots.append(ax.bxp(wichmore, positions=x2, patch_artist=True))
    names = ['Dryland Oxford trended 1972-2019', 'Winchmore 1966-2003']
    l_elements = []
    for name, bplot, c in zip(names, bplots, colors):
        l_elements.append(Patch(facecolor=c, label=name))
        for patch in bplot['boxes']:
            patch.set_facecolor(c)

    xlocs = np.arange(0.5, 24, 2)
    ax.set_xticks(ticks=np.arange(0.5, 24, 2))
    ax.set_xticklabels(['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
    ax.set_xlabel('month')
    ax.set_title('Long term pasture growth comparison')
    ax.set_ylabel('Pasture Growth kg DM/day')
    ax.set_ylim(-10, 110)
    ax.legend(handles=l_elements)
    # add vlines
    ax.vlines(xlocs[1:] - 1, ymin=-10, ymax=110, ls=':')
    fig.tight_layout()
    fig.savefig(outdir.joinpath('dryland_variability.png'))

    axs = plot_multiple_monthly_results(data=data2, out_vars=['pgr'], show=False, main_kwargs={'marker': 'o'},
                                        fig_size=fig_size)
    for v in axs.values():
        v.set_ylabel('Pasture growth kg DM/day')
        v.set_title('Mean pasture growth comparison')
        fig = v.figure
        fig.tight_layout()
        fig.savefig(outdir.joinpath('mean_production_comparison.png'))


if __name__ == '__main__':
    final_plots()
