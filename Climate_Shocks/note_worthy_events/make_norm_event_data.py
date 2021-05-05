"""
 Author: Matt Hanson
 Created: 5/05/2021 10:25 AM
 """
import pandas as pd
import numpy as np
import os
from Climate_Shocks.get_past_record import get_restriction_record, get_vcsn_record
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_smd_monthly
from BS_work.SWG.SWG_wrapper import get_monthly_smd_mean_detrended
from Climate_Shocks.climate_shocks_env import supporting_data_dir


def make_data(save, save_paths):  # todo add new system to write up, CHECK
    """
    make the final data for greg
    :param org_data: from final_event_recurance import get_org_data
    :param save:
    :param restrict_cat:
    :return:
    """
    lower_quantile, upper_quantile = 0.25, 0.75

    data = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 13), range(1972, 2020)], names=['month', 'year']),
                        columns=['temp', 'precip', 'rest', 'rest_cum'], dtype=float)

    # calc necessary variables
    vcsn = get_vcsn_record('detrended2')
    vcsn.loc[:, 'sma'] = calc_smd_monthly(vcsn.rain, vcsn.pet, vcsn.index) - vcsn.loc[:, 'doy'].replace(
        get_monthly_smd_mean_detrended(leap=False, recalc=True))
    vcsn.loc[:, 'tmean'] = (vcsn.loc[:, 'tmax'] + vcsn.loc[:, 'tmin']) / 2
    vcsn.loc[:, 'month_mapper'] = vcsn.loc[:, 'month'].astype(int)

    # make, save the cutoffs for use in checking functions!
    vcsn = vcsn.groupby(['month', 'year']).mean()
    upper_limit = vcsn.reset_index().groupby('month').quantile(upper_quantile)
    upper_limit.to_csv(os.path.join(supporting_data_dir, 'upper_limit.csv'))
    lower_limit = vcsn.reset_index().groupby('month').quantile(lower_quantile)
    lower_limit.to_csv(os.path.join(supporting_data_dir, 'lower_limit.csv'))


    rest_rec = get_restriction_record('detrended').groupby(['month', 'year']).sum().loc[:, 'f_rest']

    data.loc[:, :] = 0
    data.loc[rest_rec.index, 'rest_cum'] = rest_rec
    data = pd.merge(data, vcsn, right_index=True, left_index=True)

    # set hot
    var = 'tmean'
    idx = data.loc[:, var] >= data.loc[:, 'month_mapper'].replace(upper_limit.loc[:, var].to_dict())
    data.loc[idx, 'temp'] = 1

    # set cold
    var = 'tmean'
    idx = data.loc[:, var] <= data.loc[:, 'month_mapper'].replace(lower_limit.loc[:, var].to_dict())
    data.loc[idx, 'temp'] = -1

    # set wet
    var = 'sma'  # negative is dry positive is wet
    idx = data.loc[:, var] >= data.loc[:, 'month_mapper'].replace(upper_limit.loc[:, var].to_dict())
    data.loc[idx, 'precip'] = -1

    # set dry
    var = 'sma'  # negative is dry positive is wet
    idx = data.loc[:, var] <= data.loc[:, 'month_mapper'].replace(lower_limit.loc[:, var].to_dict())
    data.loc[idx, 'precip'] = 1

    # re-order data
    data = data.reset_index().sort_values(['year', 'month'])

    # get previous states.
    for k in ['temp', 'precip', 'rest_cum']:
        data.loc[:, 'prev_{}'.format(k)] = data.loc[:, k].shift(1)

    if save:
        for path in save_paths:
            with open(path, 'w') as f:
                f.write('0=normal temp: -1=cold; ; 1=hot; Precip: -1=wet; 1=dry; rest: 1=anomalous restrictions. '
                        'prev_{} = previous months data  \n')
            data.to_csv(path, mode='a', index=False)


if __name__ == '__main__':
    make_data(True, [])
