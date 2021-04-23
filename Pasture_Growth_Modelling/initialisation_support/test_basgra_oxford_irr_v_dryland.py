"""
 Author: Matt Hanson
 Created: 21/04/2021 1:20 PM
 """
import ksl_env
import numpy as np
import pandas as pd
from Pasture_Growth_Modelling.full_model_implementation import _gen_input, default_swg_dir, month_len, abs_max_irr, \
    out_variables, calc_pasture_growth
from Storylines.base_storylines_old import get_baseline_storyline
from Pasture_Growth_Modelling.initialisation_support.validate_dryland_v3 import make_mean_comparison, \
    get_horarata_data_old, plot_multiple_monthly_results

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from basgra_python import run_basgra_nz, get_month_day_to_nonleap_doy
from supporting_functions.output_metadata import get_output_metadata

weed_dict_2 = {
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

def compare_oxford_irr_dry(save=False):
    nsims = 100
    storyline = get_baseline_storyline()
    simlen = np.array([month_len[e] for e in storyline.month]).sum()
    params_dry, doy_irr, all_matrix_weathers, all_days_harvests_dry = _gen_input(storyline=storyline,
                                                                                 nsims=nsims, mode='dryland',
                                                                                 site='oxford',
                                                                                 chunks=1, current_c=1,
                                                                                 nperc=nsims, simlen=simlen,
                                                                                 swg_dir=default_swg_dir, fix_leap=True)

    params_irr, doy_irr, all_matrix_weathers, all_days_harvests_irr = _gen_input(storyline=storyline,
                                                                                 nsims=nsims, mode='irrigated',
                                                                                 site='oxford',
                                                                                 chunks=1, current_c=1,
                                                                                 nperc=nsims, simlen=simlen,
                                                                                 swg_dir=default_swg_dir, fix_leap=True)
    params_irr_eyre, doy_irr_eyre, all_matrix_weathers_eyre, all_days_harvests_irr_eyre = _gen_input(
        storyline=storyline,
        nsims=nsims, mode='irrigated',
        site='eyrewell',
        chunks=1, current_c=1,
        nperc=nsims, simlen=simlen,
        swg_dir=default_swg_dir, fix_leap=True)

    all_out_dry = np.zeros((len(out_variables), simlen, nsims)) * np.nan
    all_out_dry_monk = np.zeros((len(out_variables), simlen, nsims)) * np.nan
    all_out_irr = np.zeros((len(out_variables), simlen, nsims)) * np.nan
    all_out_irr_eyre = np.zeros((len(out_variables), simlen, nsims)) * np.nan
    for i, (
    matrix_weather, days_harvest_irr, days_harvest_dry, matrix_weather_eyre, days_harvest_irr_eyre) in enumerate(
            zip(all_matrix_weathers,
                all_days_harvests_irr,
                all_days_harvests_dry, all_matrix_weathers_eyre, all_days_harvests_irr_eyre)):
        if i % 10 == 0:
            print(i)
        restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
        out = run_basgra_nz(params_irr, matrix_weather, days_harvest_irr, doy_irr, verbose=False, run_365_calendar=True)
        out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
        pg = pd.DataFrame(
            calc_pasture_growth(out, days_harvest_irr, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'PGR'] = pg.loc[:, 'pg']
        out.loc[:, 'F_REST'] = restrict
        all_out_irr[:, :, i] = out.loc[:, out_variables].values.transpose()

        # run eyrewell
        pg = pd.DataFrame(
            calc_pasture_growth(out, days_harvest_irr, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'PGR'] = pg.loc[:, 'pg']
        out.loc[:, 'F_REST'] = restrict
        all_out_irr[:, :, i] = out.loc[:, out_variables].values.transpose()

        restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
        out = run_basgra_nz(params_irr_eyre, matrix_weather_eyre, days_harvest_irr_eyre, doy_irr_eyre, verbose=False,
                            run_365_calendar=True)
        out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
        pg = pd.DataFrame(
            calc_pasture_growth(out, days_harvest_irr_eyre, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'PGR'] = pg.loc[:, 'pg']
        out.loc[:, 'F_REST'] = restrict
        all_out_irr_eyre[:, :, i] = out.loc[:, out_variables].values.transpose()


        # run dryland
        matrix_weather.loc[:, 'max_irr'] = 0
        matrix_weather.loc[:, 'irr_trig'] = 0
        matrix_weather.loc[:, 'irr_targ'] = 0
        restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
        out = run_basgra_nz(params_dry, matrix_weather, days_harvest_dry, doy_irr, verbose=False, run_365_calendar=True)
        out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

        pg = pd.DataFrame(
            calc_pasture_growth(out, days_harvest_dry, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'PGR'] = pg.loc[:, 'pg']
        out.loc[:, 'F_REST'] = restrict
        all_out_dry[:, :, i] = out.loc[:, out_variables].values.transpose()

        # run dryland monk
        for m in range(1, 13):
            days_harvest_dry.loc[days_harvest_dry.index.month == m, 'weed_dm_frac'] = weed_dict_2[m]
        matrix_weather.loc[:, 'max_irr'] = 0
        matrix_weather.loc[:, 'irr_trig'] = 0
        matrix_weather.loc[:, 'irr_targ'] = 0
        restrict = 1 - matrix_weather.loc[:, 'max_irr'] / abs_max_irr
        out = run_basgra_nz(params_dry, matrix_weather, days_harvest_dry, doy_irr, verbose=False, run_365_calendar=True)
        out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

        pg = pd.DataFrame(
            calc_pasture_growth(out, days_harvest_dry, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'PGR'] = pg.loc[:, 'pg']
        out.loc[:, 'F_REST'] = restrict
        all_out_dry_monk[:, :, i] = out.loc[:, out_variables].values.transpose()

    all_out_dry = pd.DataFrame(np.nanmean(all_out_dry, axis=2).transpose(), columns=out_variables,
                               index=matrix_weather.index)
    all_out_dry_monk = pd.DataFrame(np.nanmean(all_out_dry_monk, axis=2).transpose(), columns=out_variables,
                               index=matrix_weather.index)
    all_out_irr = pd.DataFrame(np.nanmean(all_out_irr, axis=2).transpose(), columns=out_variables,
                               index=matrix_weather.index)
    all_out_irr_eyre = pd.DataFrame(np.nanmean(all_out_irr_eyre, axis=2).transpose(), columns=out_variables,
                               index=matrix_weather.index)
    if save:
        all_out_irr.to_csv("D:\mh_unbacked\SLMACC_2020\one_off_data\irr_baseline.csv")
        all_out_irr_eyre.to_csv("D:\mh_unbacked\SLMACC_2020\one_off_data\dry_baseline.csv")
        all_out_dry.to_csv("D:\mh_unbacked\SLMACC_2020\one_off_data\dry_baseline.csv")
        all_out_dry_monk.to_csv("D:\mh_unbacked\SLMACC_2020\one_off_data\dry_monk_baseline.csv")
    all_out_irr.loc[:, 'pg'] = all_out_irr.loc[:, 'PGR']
    all_out_irr_eyre.loc[:, 'pg'] = all_out_irr_eyre.loc[:, 'PGR']
    all_out_dry.loc[:, 'pg'] = all_out_dry.loc[:, 'PGR']
    all_out_dry_monk.loc[:, 'pg'] = all_out_dry_monk.loc[:, 'PGR']
    data = {

               'dryland_monkeying': all_out_dry_monk,
               'dryland_baseline': all_out_dry,
               'irrigated_oxford': all_out_irr,
               'irrigated_eyrewell':all_out_irr_eyre,

    }
    fun = 'mean'
    data2 = {e: make_mean_comparison(v, fun) for e, v in data.items()}
    data2['horata'] = get_horarata_data_old()
    plot_multiple_monthly_results(data=data2, out_vars=['pg', 'pgr'], show=True, main_kwargs={'marker': 'o'})


if __name__ == '__main__':
    compare_oxford_irr_dry(False)  # todo check and debug!, multiple years showing....  # tod start here!  # todo why cant we just normalise to historical data??? e.g. v3?
    # todo is this the problem of year 1 not being close to year 2 etc.
