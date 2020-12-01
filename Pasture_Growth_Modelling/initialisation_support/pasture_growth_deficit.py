"""
 Author: Matt Hanson
 Created: 2/12/2020 12:08 PM
 """
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth
from Pasture_Growth_Modelling.initialisation_support.inital_long_term_runs import run_past_basgra_irrigated, \
    run_past_basgra_dryland
import pandas as pd


def calc_pasture_growth_anomaly(mode='irrigated'):
    if mode == 'irrigated':
        out, (params, doy_irr, matrix_weather, days_harvest) = run_past_basgra_irrigated(True)
    elif mode == 'dryland':
        out, (params, doy_irr, matrix_weather, days_harvest) = run_past_basgra_dryland(True)
    else:
        raise ValueError('unexpected mode')

    pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, 'from_yeild_regular', '10D', 'mean'))

    pg.loc[:, 'doy'] = pd.Series(pg.index, index=pg.index).dt.dayofyear
    pg.loc[:, 'pg_normal'] = pg.loc[:, 'doy']
    mapper = pg.groupby('doy').mean().loc[:, 'pg'].to_dict()
    pg = pg.replace({'pg_normal': mapper})
    pg.loc[:, 'pga'] = pg.loc[:, 'pg'] - pg.loc[:, 'pg_normal']
    pg.loc[:, 'pga_norm'] = pg.loc[:, 'pga'] / pg.loc[:, 'pg_normal']

    return pg


if __name__ == '__main__':
    test = calc_pasture_growth_anomaly()
    pass
