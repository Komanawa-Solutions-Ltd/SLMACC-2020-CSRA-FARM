"""
 Author: Matt Hanson
 Created: 23/04/2021 1:37 PM
 """
import ksl_env
import numpy as np
import datetime
import pandas as pd
import os
from Climate_Shocks import climate_shocks_env
from Climate_Shocks.get_past_record import get_vcsn_record, get_restriction_record
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth, calc_pasture_growth_anomaly

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from basgra_python import run_basgra_nz


# todo baseline trended, detrended, or average from historical quantified, trended/detrened?
def run_past_basgra_irrigated(return_inputs=False, site='eyrewell', reseed=True, version='trended'):
    mode = 'irrigated'
    print('running: {}, {}, reseed: {}'.format(mode, site, reseed))
    weather = get_vcsn_record(version=version, site=site)
    rest = get_restriction_record(version=version)
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest, fix_leap=False)
    days_harvest = create_days_harvest(mode, matrix_weather, site, fix_leap=False)

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


def run_past_basgra_dryland(return_inputs=False, site='eyrewell', reseed=True, version='trended'):
    mode = 'dryland'
    print('running: {}, {}, reseed: {}'.format(mode, site, reseed))
    weather = get_vcsn_record(site=site, version=version)
    rest = None
    params, doy_irr = get_params_doy_irr(mode)
    matrix_weather = create_matrix_weather(mode, weather, rest, fix_leap=False)
    days_harvest = create_days_harvest(mode, matrix_weather, site, fix_leap=False)
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


if __name__ == '__main__':
    t, rd = get_historical_average_baseline('eyrewell', 'irrigated', [2024], 'PGR', version='trended')
    t2, rd = get_historical_average_baseline('eyrewell', 'irrigated', [2024], 'PGR', version='detrended2')
    import matplotlib.pyplot as plt

    plt.plot(t.month, t.PGR, label='trended')
    plt.plot(t2.month, t2.PGR, label='detrended')
    plt.legend()
    plt.show()
    pass
