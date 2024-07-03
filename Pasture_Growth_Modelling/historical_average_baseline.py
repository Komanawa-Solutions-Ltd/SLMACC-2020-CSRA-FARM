"""
 Author: Matt Hanson
 Created: 23/04/2021 1:37 PM
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
from Storylines.storyline_building_support import month_len
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth, calc_pasture_growth_anomaly
from Pasture_Growth_Modelling.basgra_parameter_sets import default_mode_sites

# add basgra nz functions
from komanawa.basgra_nz_py import run_basgra_nz


def run_past_basgra_irrigated(return_inputs=False, site='eyrewell', reseed=True, version='trended', mode='irrigated'):
    print('running: {}, {}, reseed: {}'.format(mode, site, reseed))
    weather = get_vcsn_record(version=version, site=site)
    rest = get_restriction_record(version=version)
    params, doy_irr = get_params_doy_irr(mode)
    all_out = []
    for y in range(1972, 2019):
        temp_weather = weather.loc[(weather.index>=f'{y}-07-01') & (weather.index<f'{y+1}-07-01')]
        temp_rest = rest.loc[(rest.index>=f'{y}-07-01') & (rest.index<f'{y+1}-07-01')]

        matrix_weather = create_matrix_weather(mode, temp_weather, temp_rest, fix_leap=False)
        days_harvest = create_days_harvest(mode, matrix_weather, site, fix_leap=False)

        if not reseed:
            days_harvest.loc[:, 'reseed_trig'] = -1

        out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
        out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

        pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'pg'] = pg.loc[:, 'pg']
        all_out.append(out)
    all_out = pd.concat(all_out)
    all_out = calc_pasture_growth_anomaly(all_out, fun='mean')

    if return_inputs:
        return all_out, (params, doy_irr, matrix_weather, days_harvest)
    return all_out


def run_past_basgra_dryland(return_inputs=False, site='eyrewell', reseed=True, version='trended'):
    mode = 'dryland'
    print('running: {}, {}, reseed: {}'.format(mode, site, reseed))
    weather = get_vcsn_record(site=site, version=version)
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    all_out = []
    for y in range(1972, 2019):
        temp_weather = weather.loc[(weather.index>=f'{y}-07-01') & (weather.index<f'{y+1}-07-01')]

        matrix_weather = create_matrix_weather(mode, temp_weather, rest, fix_leap=False)
        days_harvest = create_days_harvest(mode, matrix_weather, site, fix_leap=False)
        if not reseed:
            days_harvest.loc[:, 'reseed_trig'] = -1

        out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
        out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
        pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'pg'] = pg.loc[:, 'pg']
        all_out.append(out)
    all_out = pd.concat(all_out)
    all_out = calc_pasture_growth_anomaly(all_out, fun='mean')

    if return_inputs:
        return all_out, (params, doy_irr, matrix_weather, days_harvest)
    return all_out


def get_historical_average_baseline(site, mode, years, key='PGR', recalc=False, version='trended'):
    """
    get the historical average baseline data.  Do I assue a July start? or return the whole year
    :param site:
    :param mode:
    :param years: list of years (e.g. 2024, 2025)
    :param recalc: bool if True recalc the data
    :return:
    """
    save_path = os.path.join(climate_shocks_env.supporting_data_dir,
                             'baseline_data', f'historical_average-{site}-{mode}-{version}.csv')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if os.path.exists(save_path) and not recalc:
        with open(save_path, 'r') as f:
            run_date = f.readline().strip()
        out = pd.read_csv(save_path, skiprows=1)
    else:
        if site == 'oxford' and mode == 'dryland':
            out = run_past_basgra_dryland(return_inputs=False, site='oxford', reseed=True, version=version)
        elif site == 'oxford' and (mode == 'irrigated' or 'store' in mode):
            out = run_past_basgra_irrigated(site='oxford', version=version, mode=mode)
        elif site == 'eyrewell' and (mode == 'irrigated' or 'store' in mode):
            out = run_past_basgra_irrigated(site='eyrewell', version=version, mode=mode)
        else:
            raise ValueError(f'wierd values for site,mode {site}-{mode}')
        run_date = datetime.datetime.now().isoformat()
        with open(save_path, 'w') as f:
            f.write(f'{run_date}\n')
        out.to_csv(save_path, mode='a')

    out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    if key == 'PGR':
        all_data = out.groupby('month').mean()['pg'].to_dict()
    elif key in ['PGRA', 'PGRA_cum', 'F_REST']:
        all_data = None
    else:
        all_data = out.groupby('month').mean()[key].to_dict()

    outdata = []
    for y in years:
        temp = pd.DataFrame(index=range(0, 365), columns=['month', 'year', 'doy', key])
        temp.loc[:, 'doy'] = np.arange(1, 366)
        temp.loc[:, 'year'] = y
        temp.loc[:, 'month'] = month = pd.to_datetime([f'2025-{d}' for d in np.arange(1, 366)],
                                                      format='%Y-%j').month.values
        if key in ['PGRA', 'PGRA_cum', 'F_REST']:
            temp.loc[:, key] = np.nan
        else:
            temp.loc[:, key] = [all_data[m] for m in month]
        outdata.append(temp)

    outdata = pd.concat(outdata)

    return outdata, run_date

def get_historical_median_baseline(site, mode, years, key='PGR', recalc=False, version='trended'):
    """
    get the historical average baseline data.  Do I assue a July start? or return the whole year
    :param site:
    :param mode:
    :param years: list of years (e.g. 2024, 2025)
    :param recalc: bool if True recalc the data
    :return:
    """
    save_path = os.path.join(climate_shocks_env.supporting_data_dir,
                             'baseline_data', f'historical_average-{site}-{mode}-{version}.csv')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if os.path.exists(save_path) and not recalc:
        with open(save_path, 'r') as f:
            run_date = f.readline().strip()
        out = pd.read_csv(save_path, skiprows=1)
    else:
        if site == 'oxford' and mode == 'dryland':
            out = run_past_basgra_dryland(return_inputs=False, site='oxford', reseed=True, version=version)
        elif site == 'oxford' and mode == 'irrigated':
            out = run_past_basgra_irrigated(site='oxford', version=version)
        elif site == 'eyrewell' and mode == 'irrigated':
            out = run_past_basgra_irrigated(site='eyrewell', version=version)
        else:
            raise ValueError(f'wierd values for site,mode {site}-{mode}')
        run_date = datetime.datetime.now().isoformat()
        with open(save_path, 'w') as f:
            f.write(f'{run_date}\n')
        out.to_csv(save_path, mode='a')

    out.loc[:, 'PER_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']
    if key == 'PGR':
        all_data = out.groupby('month').median()['pg'].to_dict()
    elif key in ['PGRA', 'PGRA_cum', 'F_REST']:
        all_data = None
    else:
        all_data = out.groupby('month').median()[key].to_dict()

    outdata = []
    for y in years:
        temp = pd.DataFrame(index=range(0, 365), columns=['month', 'year', 'doy', key])
        temp.loc[:, 'doy'] = np.arange(1, 366)
        temp.loc[:, 'year'] = y
        temp.loc[:, 'month'] = month = pd.to_datetime([f'2025-{d}' for d in np.arange(1, 366)],
                                                      format='%Y-%j').month.values
        if key in ['PGRA', 'PGRA_cum', 'F_REST']:
            temp.loc[:, key] = np.nan
        else:
            temp.loc[:, key] = [all_data[m] for m in month]
        outdata.append(temp)

    outdata = pd.concat(outdata)

    return outdata, run_date


def export_true_historical():
    outdir = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'true_historical_average_trended')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    outdata = pd.DataFrame(index=[7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 'total'])
    outdata.index.name = 'month'
    for site, mode in zip(['eyrewell', 'oxford', 'oxford'], ['irrigated', 'irrigated', 'dryland']):
        t, rd = get_historical_average_baseline(site, mode, [2024], 'PGR', version='trended')
        t = t.loc[:, ['month', 'PGR']].drop_duplicates().set_index('month')
        outdata.loc[months, f'{site}-{mode}'] = t.loc[months, 'PGR']
        outdata.loc['total', f'{site}-{mode}'] = np.sum([t.loc[m, 'PGR'] * month_len[m] for m in months])
    outdata.to_csv(os.path.join(outdir, 'baseline_scen_pg.csv'))


if __name__ == '__main__':
    for mode, site in default_mode_sites:
        t, rd = get_historical_average_baseline(site, mode, years=[2024], recalc=False)

    old = False
    if old:
        for v in ['trended']: #no detrended for oxford...
           t, rd = get_historical_average_baseline('eyrewell', 'irrigated', [2024], 'PGR', version=v, recalc=True)
           t, rd = get_historical_average_baseline('oxford', 'irrigated', [2024], 'PGR', version=v, recalc=True)
           t, rd = get_historical_average_baseline('oxford', 'dryland', [2024], 'PGR', version=v, recalc=True)

        t, rd = get_historical_average_baseline('eyrewell', 'irrigated', [2024], 'PGR', version='trended')
        t2, rd = get_historical_average_baseline('eyrewell', 'irrigated', [2024], 'PGR', version='detrended2')
        import matplotlib.pyplot as plt

        plt.plot(t.month, t.PGR, label='trended')
        plt.plot(t2.month, t2.PGR, label='detrended')
        plt.legend()
        plt.show()
        export_true_historical()
        pass
