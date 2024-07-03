"""
 Author: Matt Hanson
 Created: 9/03/2021 10:44 AM
 """
import numpy as np
import os
from Climate_Shocks import climate_shocks_env
import itertools
import pandas as pd
from Storylines.storyline_params import irrig_season



def fix_precip(x):
    if x == 1:
        return "D"
    else:
        if not np.isnan(x):
            return "ND"

default_rest_dir = os.path.join(climate_shocks_env.supporting_data_dir, 'rest_mapper')
default_rest_path = os.path.join(climate_shocks_env.supporting_data_dir,
                                         'restriction_record_detrend.csv')


def get_irr_by_quantile(recalc=False, outdir=default_rest_dir, rest_path=default_rest_path):
    dnd = [['D', 'ND'], ['D', 'ND']]
    possible_quantiles = np.arange(1, 100) / 100

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if os.listdir(outdir) == [f'{m1}-{m2}_rest.csv' for m1, m2 in itertools.product(*dnd)] and not recalc:
        out = {}
        for m1, m2 in itertools.product(*dnd):
            quantile = pd.read_csv(os.path.join(outdir, f'{m1}-{m2}_rest.csv'), index_col=0)
            quantile.columns = quantile.columns.astype(int)
            temp = quantile.values
            temp[np.isclose(temp,0)] = 0.00000001  # to ensure that proabilities are calculated
            out[f'{m1}-{m2}'] = quantile
        return out

    rest_data = pd.read_csv(rest_path)
    rest_data.drop(columns='date', inplace=True)
    rest_data = rest_data.groupby(['year', 'month']).mean()
    event_data = pd.read_csv(climate_shocks_env.event_def_path, skiprows=1)
    event_data = event_data.set_index(['year', 'month'])
    event_data.loc[:, 'rest'] = rest_data.loc[:, 'f_rest']
    event_data.loc[:, 'dnd'] = [fix_precip(e) for e in event_data.loc[:, 'precip']]
    event_data.loc[:, 'prev_dnd'] = [fix_precip(e) for e in event_data.loc[:, 'prev_precip']]

    out = {}
    for m1, m2 in itertools.product(*dnd):
        outdata = pd.DataFrame(index=possible_quantiles, columns=irrig_season, dtype=float)
        for m in irrig_season:
            temp = event_data.loc[:, m, :]
            temp = temp.loc[(temp.dnd == m2) & (temp.prev_dnd == m1)]
            outdata.loc[:, m] = temp.loc[:, 'rest'].quantile(possible_quantiles).fillna(0)
        out[f'{m1}-{m2}'] = outdata
        temp = outdata.values
        temp[np.isclose(temp,0)] = 0.00000001  # to ensure that proabilities are calculated
        outdata.to_csv(os.path.join(outdir, f'{m1}-{m2}_rest.csv'))
    return out

if __name__ == '__main__':
    t = get_irr_by_quantile(False)
    pass