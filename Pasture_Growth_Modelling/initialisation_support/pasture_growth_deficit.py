"""
 Author: Matt Hanson
 Created: 2/12/2020 12:08 PM
 """
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth
from Pasture_Growth_Modelling.initialisation_support.inital_long_term_runs import run_past_basgra_irrigated, \
    run_past_basgra_dryland
import pandas as pd


def calc_past_pasture_growth_anomaly(mode='irrigated', pg_mode='from_yield', freq='1D', fun='mean', site='oxford'):
    if mode == 'irrigated':
        out, (params, doy_irr, matrix_weather, days_harvest) = run_past_basgra_irrigated(True, site=site)
    elif mode == 'dryland':
        out, (params, doy_irr, matrix_weather, days_harvest) = run_past_basgra_dryland(True, site=site)
    else:
        raise ValueError('unexpected mode')

    pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, pg_mode, freq, fun))

    pg.loc[:, 'month'] = pd.Series(pg.index, index=pg.index).dt.month
    pg.loc[:, 'pg_normal'] = pg.loc[:, 'month']
    mapper = pg.groupby('month').mean().loc[:, 'pg'].to_dict() #todo make this the same function
    pg = pg.replace({'pg_normal': mapper})
    pg.loc[:, 'pga'] = pg.loc[:, 'pg'] - pg.loc[:, 'pg_normal']
    pg.loc[:, 'pga_norm'] = pg.loc[:, 'pga'] / pg.loc[:, 'pg_normal']

    return pg


if __name__ == '__main__':
    test = calc_past_pasture_growth_anomaly('dryland', pg_mode='from_dmh', freq='month', fun='mean').reset_index()
    test.plot('date', ['pg_normal'])
    import matplotlib.pyplot as plt
    plt.show()
    pass
