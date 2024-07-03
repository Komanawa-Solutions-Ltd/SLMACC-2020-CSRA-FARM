"""
 Author: Matt Hanson
 Created: 13/10/2021 8:48 AM
 """
import project_base
import numpy as np
import datetime
import pandas as pd
import os
from Climate_Shocks import climate_shocks_env
from Climate_Shocks.get_past_record import get_vcsn_record, get_restriction_record
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather
import matplotlib.pyplot as plt
from Storylines.storyline_building_support import month_len
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth, calc_pasture_growth_anomaly
from Pasture_Growth_Modelling.basgra_parameter_sets import default_mode_sites
from Pasture_Growth_Modelling.historical_average_baseline import run_past_basgra_irrigated
from Pasture_Growth_Modelling.storage_parameter_sets import site_mode_dep_params

# add basgra nz functions
from komanawa.basgra_nz_py.basgra_python import run_basgra_nz
from komanawa.basgra_nz_py.supporting_functions.plotting import plot_multiple_results, plot_multiple_monthly_results, \
    plot_multiple_monthly_violin_box, plot_multiple_date_range
from komanawa.basgra_nz_py.supporting_functions.cumulative_annual_data import make_cumuliative_annual
from komanawa.basgra_nz_py.supporting_functions.output_metadata import get_output_metadata

out_variables = (
    'BASAL',  # should some of these be amalgamated to sum, no you can multiply against # of days in the month.
    'PGR',
    'PER_PAW',
    'F_REST',
    'IRRIG',
    'IRRIG_DEM',
    'RESEEDED',
    'DMH_RYE',
    'YIELD',
    'irrig_store',
    'irrig_scheme',
    'h2o_store_vol',
    'h2o_store_per_area',
    'store_scheme_in',
    'store_overflow'

)

jun_params = [
    'LAI',
    'TILG2',
    'TILG1',
    'TILV',
    'CLV',
    'CRES',
    'CST',
    'CSTUB',
    'BASAL',
]

jul_params = [
    'CLV',
    'CRES',
    'CRT',
]


def produce_storage_initialisation_and_checks(outdir, plot=False):
    os.makedirs(outdir, exist_ok=True)
    version = 'trended'
    oxford_sites = {}
    eyrewell_sites = {}
    monthly_oxford_sites = {}
    monthly_eyrewell_sites = {}
    outdata = pd.DataFrame(index=jun_params + jul_params)
    reseed_basal_data = pd.DataFrame(index=range(1972, 2020))
    reseed_date_path = os.path.join(outdir, 'reseed_dates.txt')
    with open(reseed_date_path, 'w') as f:
        f.write('reseed dates for the full suite\n')
    for mode, site in default_mode_sites:
        if mode == 'dryland':
            continue
        weather = get_vcsn_record(version=version, site=site)
        rest = get_restriction_record(version=version)
        params, doy_irr = get_params_doy_irr(mode)
        matrix_weather = create_matrix_weather(mode, weather, rest, fix_leap=False)
        days_harvest = create_days_harvest(mode, matrix_weather, site, fix_leap=False)

        out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
        out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
        out.loc[:, 'F_REST'] = rest.f_rest

        # cumulative storage input and irrigation applied and stored irrigation
        new_out_vars = []
        cum_vars = [
            'F_REST',
            'IRRIG',
            'irrig_scheme',
            'irrig_store',
            'store_scheme_in',
            'store_overflow',

        ]
        for k in cum_vars:
            out.loc[:, f'cum_{k}'] = make_cumuliative_annual(out.loc[:, k], 1, 7)
            new_out_vars.append(f'cum_{k}')

        pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'PGR'] = pg.loc[:, 'pg']
        out.to_csv(os.path.join(outdir, f'{site}-{mode}-full.csv'))

        with open(reseed_date_path, 'a') as f:
            f.write(f'\n\n{site}-{mode} years:\n')
            f.write('\n'.join(out.loc[out.RESEEDED > 0, 'year'].values.astype(str)))

        temp = out.loc[out.index.month == 6, jun_params].mean()
        temp.loc['BASAL'] *= 1 / 100
        outdata.loc[temp.index, f'{site}-{mode}'] = temp
        temp = out.loc[out.index.month == 7, jul_params].mean()
        outdata.loc[temp.index, f'{site}-{mode}'] = temp
        out.loc[:, 'month'] = out.index.month
        if mode != 'dryland':
            if site == 'oxford':
                oxford_sites[mode] = out
                monthly_oxford_sites[mode] = out.groupby('month').mean()
            elif site == 'eyrewell':
                eyrewell_sites[mode] = out
                monthly_eyrewell_sites[mode] = out.groupby('month').mean()
            else:
                raise ValueError("shouldn't get here")
        reseed_basal_data.loc[out.loc[out.doy == 151, 'year'].astype(int), f'{site}-{mode}'] = out.loc[
            out.doy == 151, 'BASAL'].values
    reseed_basal_data.to_csv(os.path.join(outdir, 'reseed_basals.csv'))
    outdata.to_csv(os.path.join(outdir, 'initial_params.csv'))

    # plotting
    if plot:
        out_variables_use = list(out_variables) + new_out_vars
        for k, d, md in zip(('eyrewell', 'oxford'), (eyrewell_sites, oxford_sites),
                            (monthly_eyrewell_sites, monthly_oxford_sites)):
            plot_outdir = os.path.join(outdir, k)

            plot_multiple_monthly_violin_box(data=d, outdir=os.path.join(plot_outdir, 'monthly_box'),
                                             out_vars=out_variables_use,
                                             fig_size=(10, 8), title_str=k,
                                             main_kwargs={}, label_main=True, show=False)
            plt.close('all')
            plot_multiple_monthly_results(data=md, outdir=os.path.join(plot_outdir, 'monthly'), out_vars=out_variables_use,
                                          fig_size=(10, 8), title_str=k,
                                          main_kwargs={}, label_main=True, show=False)
            plt.close('all')
            plot_multiple_results(data=d, outdir=os.path.join(plot_outdir, 'full_suite'), out_vars=out_variables_use,
                                  fig_size=(10, 8), title_str=k,
                                  rolling=10, main_kwargs={'alpha': 0}, rolling_kwargs={}, label_rolling=True,
                                  label_main=False,
                                  show=False)
            plt.close('all')
            for y in range(1972, 2021, 2):
                plot_multiple_date_range(data=d, start_date=f'{y}-07-01', end_date=f'{y + 2}-06-30',
                                         outdir=os.path.join(plot_outdir, 'full_suite_2yrs'), out_vars=out_variables_use,
                                         fig_size=(10, 8), title_str=k,
                                         rolling=10, main_kwargs={'alpha': 0}, rolling_kwargs={}, label_rolling=True,
                                         label_main=False,
                                         show=False)
                plt.close('all')


def test_storage_initialisation_and_checks(outdir):
    os.makedirs(outdir, exist_ok=True)
    for site, mode in site_mode_dep_params.keys():
        out = run_past_basgra_irrigated(site=site, mode=mode)


if __name__ == '__main__':
    produce_storage_initialisation_and_checks(os.path.join(ksl_env.slmmac_dir, r"outputs_for_ws\norm",
                                                           'storage_initalisation_to_WS'),
                                              plot=True)
