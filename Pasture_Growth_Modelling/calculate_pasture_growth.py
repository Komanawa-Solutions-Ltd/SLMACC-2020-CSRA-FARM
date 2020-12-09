"""
 Author: Matt Hanson
 Created: 2/12/2020 9:32 AM
 """

import pandas as pd
import numpy as np



def calc_pasture_growth_anomaly(basgra_out, an_per='month', fun='mean'):
    if an_per=='month':
        basgra_out.loc[:, 'month'] = pd.Series(basgra_out.index, index=basgra_out.index).dt.month
        basgra_out.loc[:, 'pg_normal'] = basgra_out.loc[:, 'month']
        mapper = basgra_out.groupby('month').agg({'pg': fun}).loc[:, 'pg'].to_dict()
        basgra_out = basgra_out.replace({'pg_normal': mapper})
        basgra_out.loc[:, 'pga'] = basgra_out.loc[:, 'pg'] - basgra_out.loc[:, 'pg_normal']
        basgra_out.loc[:, 'pga_norm'] = basgra_out.loc[:, 'pga'] / basgra_out.loc[:, 'pg_normal']
    else:
        raise NotImplementedError

    return basgra_out



def calc_pasture_growth(basgra_out, basgra_harvest, mode, freq, resamp_fun):
    """
    calculate pasture growth
    :param basgra_out: the outputs from BASGRA (with datetime date index, standard with basgra output)
    :param basgra_harvest: input harvest data with date datetime index
    :param mode: allows multiple versions
    :param freq: the frequency to return data, month does a groupby on month
    :param resamp_fun: function to resample on
    :return: time series of pasture growth
    """
    # todo test to esure that frequency is greather than or equal daily

    mode_dict = {
        'from_yield': _calc_pasture_from_yeild_regular,
        'from_dmh': _calc_pasture_from_dhm,

    }
    if mode in mode_dict.keys():
        out = mode_dict[mode](basgra_out, basgra_harvest, freq, resamp_fun)
    else:
        raise ValueError('unexpected mode: {},\nexpected one of:{}'.format(mode, mode_dict.keys()))

    return out


def _resample_growth(in_data, ts_start, ts_stop, freq, resamp_fun):
    """

    :param in_data: pd.Series of cumulative growth in kg/m2, days can be missing
    :param ts_start: start date of the time series
    :param ts_stop: the stop date of the time series
    :param freq: frequency code, as per pd.Date_Range expect
    :return:
    """

    # make daily data averaged from input data
    in_data = pd.DataFrame(in_data.rename('pg_org'))
    out = pd.DataFrame(index=pd.date_range(ts_start, ts_stop, name='date'))
    out.loc[:, 'pg_org'] = np.nan
    out = out.combine_first(in_data)
    out.loc[:, 'pg'] = out.loc[:, 'pg_org']
    out.loc[:, 'pg'] = out.loc[:, 'pg'].fillna(method='bfill')
    out.loc[:, 'period_id'] = pd.notna(out.loc[:, 'pg_org']).astype(
        int).cumsum() - 99999999  # to ensure able to replace
    out.loc[pd.notna(out.loc[:, 'pg_org']), 'period_id'] += -1
    replace_dict = out.groupby('period_id').count().loc[:, 'pg'].to_dict()
    out.loc[:, 'ndays'] = out.loc[:, 'period_id']
    out = out.replace({'ndays': replace_dict})
    out.loc[:, 'pg'] *= 1 / out.loc[:, 'ndays']

    # make month and year
    out.loc[:, 'month'] = out.index.month
    out.loc[:, 'year'] = out.index.year
    # resample
    if freq == '1D':
        out = out.loc[:, 'pg']
    elif freq == 'month':
        out = out.groupby(['year', 'month']).agg({'pg': resamp_fun}).reset_index()
        strs = ['{}-{:02d}-01'.format(y, m) for y, m in out.loc[:, ['year', 'month']].itertuples(False, None)]
        out.loc[:, 'date'] = pd.to_datetime(strs)
        out = out.set_index('date').loc[:, 'pg']
    elif freq == 'year':
        out = out.groupby(['year']).agg({'pg': resamp_fun}).reset_index()
        strs = ['{}-01-01'.format(y) for y in out.loc[:, 'year'].itertuples(False, None)]
        out.loc[:, 'date'] = pd.to_datetime(strs)
        out = out.set_index('date').loc[:, 'pg']

    else:
        out = out.loc[:, 'pg'].resample(freq).agg(resamp_fun)

    return out


def _calc_pasture_from_yeild_regular(basgra_out, basgra_harvest, freq, resamp_fun):
    pg = pd.merge(basgra_out, basgra_harvest, right_index=True, left_index=True).loc[:, ['YIELD', 'harv_trig']]
    pg.loc[:, 'yield_dif'] = pg.loc[:, 'YIELD'].diff(1) * 1000  # convert to kg/m2
    pg.loc[pg.loc[:, 'yield_dif'] < 0, 'yield_dif'] = 0

    idx = (pg.loc[:, 'harv_trig'] > 0)
    out = _resample_growth(pg.loc[idx, 'yield_dif'],
                           ts_start=pg.index.min(),
                           ts_stop=pg.index.max(), freq=freq,
                           resamp_fun=resamp_fun)
    return out

def _calc_pasture_from_dhm(basgra_out, basgra_harvest, freq, resamp_fun):
    # basgra harvest kept to keep all of the inputs for these methods identical
    pg = basgra_out.copy(deep=True)
    pg.loc[:, 'dmh_dif'] = pg.loc[:, 'DMH'].diff(1)

    # add back in the removed data when it is greater than 0
    rmed = (pg.loc[:,'DM_RYE_RM'] + pg.loc[:,'DM_WEED_RM']).values[0:-1]
    pg.loc[:,'dmh_dif'] += np.concatenate([[0],rmed])

    out = _resample_growth(pg.loc[:, 'dmh_dif'],
                           ts_start=pg.index.min(),
                           ts_stop=pg.index.max(), freq=freq,
                           resamp_fun=resamp_fun)
    return out


if __name__ == '__main__':
    from Pasture_Growth_Modelling.initialisation_support.inital_long_term_runs import run_past_basgra_irrigated

    out, (params, doy_irr, matrix_weather, days_harvest) = run_past_basgra_irrigated(True)
    test = calc_pasture_growth(out, days_harvest, mode='from_yeild', freq='10D', resamp_fun='mean')
    pass
