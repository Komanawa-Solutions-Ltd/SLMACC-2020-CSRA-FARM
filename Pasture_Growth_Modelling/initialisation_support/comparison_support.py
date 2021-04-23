"""
 Author: Matt Hanson
 Created: 10/12/2020 12:27 PM
 """
import numpy as np
import ksl_env

import pandas as pd

ndays = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def make_mean_comparison(out, fun):
    out.loc[:, 'month'] = out.index.month
    out_norm = out.groupby('month').agg({'pg': fun}).to_dict()

    out_sum = pd.DataFrame(index=pd.date_range('2011-01-01', '2011-12-31', name='date'), columns=['pg'], dtype=int)
    out_sum.loc[:, 'pg'] = pd.to_numeric(out_sum.index.month)
    out_sum = out_sum.replace(out_norm)
    out_sum.loc[:, 'month'] = out_sum.index.month
    out_sum = pd.DataFrame(out_sum.groupby('month').sum().loc[:, 'pg'])
    out_sum.loc[:, 'pgr'] = out_sum.loc[:, 'pg'] / [ndays[e] for e in out_sum.index]

    return out_sum


def make_mean_comparison_suite(out, fun):
    out.loc[:, 'month'] = out.index.month
    out_norm = out.groupby(['month', 'year']).agg({'pg': fun}).reset_index().astype(float)
    months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    out = {'pg': [[] for e in range(12)], 'pgr': [[] for e in range(12)]}
    for i, m in enumerate(months):
        out['pg'][i].extend(out_norm.loc[out_norm.month == m, 'pg'] * ndays[m])
        out['pgr'][i].extend(out_norm.loc[out_norm.month == m, 'pg'])

    return out

def make_total_suite(out):
    out.loc[:, 'month'] = out.index.month
    out.loc[:,'pg'] = [e for e,m in out.loc[:,['pg','month']].itertuples(False,None)]
    out_norm = out.groupby(['year']).agg({'pg': 'sum'}).reset_index().astype(float)
    return out_norm



def get_horarata_data():
    out = pd.read_csv(
        ksl_env.shared_drives(r"Z2003_SLMACC\pasture_growth_modelling\dryland tuning\hororata_dryland.csv"))
    out.loc[:, 'date'] = pd.to_datetime(out.loc[:, 'date'])
    out.loc[:, 'doy'] = out.loc[:, 'date'].dt.dayofyear
    out.set_index('doy', inplace=True)

    full_out = pd.DataFrame(index=range(1, (367) * 3), columns=['pg'])

    idx = out.index.values
    full_out.loc[idx, 'pg'] = out.loc[:, 'pg'].values
    idx += 365
    full_out.loc[idx, 'pg'] = out.loc[:, 'pg'].values
    full_out.loc[idx + 365, 'pg'] = out.loc[:, 'pg'].values
    full_out.loc[:, 'pg'] = pd.to_numeric(full_out.loc[:, 'pg'])
    full_out.loc[:, 'pg'] = full_out.loc[:, 'pg'].interpolate(method='linear')

    idx = np.arange(1, 367)
    full_out = full_out.loc[idx + 365]
    full_out.loc[:, 'doy'] = idx
    full_out.set_index('doy', inplace=True)

    return full_out


def get_witchmore():
    # this was read from plots in Comparison of outputs of a biophysical simulation
    # model for pasture growth and composition with
    # measured data under dryland and irrigated
    # conditions in New Zealand
    out = pd.DataFrame(index=(range(12)))
    out.loc[:, 'month'] = range(1, 13)
    out.loc[:, 'pgr'] = [15, 10, 12, 10, 10, 10, 10, 10, 30, 60, 40, 19]
    out.loc[:, 'pg'] = out.loc[:, 'pgr'] * [ndays[e] for e in out.month]
    out.set_index('month', inplace=True)
    return out


def get_horarata_data_old():
    out = pd.read_csv(
        ksl_env.shared_drives(r"Z2003_SLMACC\pasture_growth_modelling\dryland tuning\hororata_dryland.csv"))
    out.loc[:, 'date'] = pd.to_datetime(out.loc[:, 'date'])
    out.loc[:, 'month'] = out.loc[:, 'date'].dt.month
    out.set_index('date', inplace=True)

    out_sum = pd.DataFrame(index=pd.date_range('2011-01-01', '2011-12-31'), columns=['pg'], dtype=float)
    out_sum.loc[out.index, 'pg'] = out.loc[:, 'pg'].values

    temp = pd.concat([out_sum, out_sum, out_sum]).reset_index().loc[:, 'pg'].interpolate(method='linear').values
    out_sum.loc[:, 'pg'] = temp[366:366 + 365]
    out_sum.loc[:, 'month'] = out_sum.index.month
    out_sum = pd.DataFrame(out_sum.groupby('month').sum().loc[:, 'pg'])
    out_sum.loc[:, 'pgr'] = out_sum.loc[:, 'pg'] / [ndays[e] for e in out_sum.index]

    return out_sum


def get_indicative_irrigated():
    data = pd.read_csv(ksl_env.shared_drives(r"Z2003_SLMACC\pasture_growth_modelling\SamSBPastureGrowth_irrigated.csv"),
                       index_col=0).to_dict()
    out_sum = pd.DataFrame(index=pd.date_range('2011-01-01', '2011-12-31', name='date'), columns=['pg'], dtype=int)
    out_sum.loc[:, 'pg'] = pd.to_numeric(out_sum.index.month)
    out_sum = out_sum.replace(data)
    out_sum.loc[:, 'month'] = out_sum.index.month
    out_sum = pd.DataFrame(out_sum.groupby('month').sum().loc[:, 'pg'])
    out_sum.loc[:, 'pgr'] = out_sum.loc[:, 'pg'] / [ndays[e] for e in out_sum.index]

    return out_sum
