"""
 Author: Matt Hanson
 Created: 21/12/2020 8:47 AM
 """
import pandas as pd
import numpy as np
import os
import project_base
from Climate_Shocks.get_past_record import get_restriction_record
from Climate_Shocks.climate_shocks_env import event_def_path


outdir = project_base.slmmac_dir.joinpath("event_definition/mutual_info")
if not os.path.exists(outdir):
    os.makedirs(outdir)

event_months = {
    'hot': [11, 12, 1, 2, 3],
    'cold': list(range(5, 10)),
    'dry': list(range(8, 13)) + list(range(1, 6)),
    'wet': list(range(5, 10)),
    'rest': list(range(9, 13)) + list(range(1, 5))}


def make_data(org_data, save=False, restrict_cat=True, save_paths=(event_def_path,)):
    """
    make the final data for greg
    :param org_data: from final_event_recurance import get_org_data
    :param save:
    :param restrict_cat:
    :return:
    """
    data = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 13), range(1972, 2020)], names=['month', 'year']),
                        columns=['temp', 'precip', 'rest', 'rest_cum'], dtype=float)
    rest_rec = get_restriction_record().groupby(['month', 'year']).sum().loc[:, 'f_rest']

    data.loc[:, :] = 0
    data.loc[rest_rec.index, 'rest_cum'] = rest_rec
    data.loc[org_data.loc[pd.notna(org_data.hot)].hot, 'temp'] = 1
    data.loc[org_data.loc[pd.notna(org_data.cold)].cold, 'temp'] = -1
    data.loc[org_data.loc[pd.notna(org_data.dry)].dry, 'precip'] = 1
    data.loc[org_data.loc[pd.notna(org_data.wet)].wet, 'precip'] = -1
    data.loc[org_data.loc[pd.notna(org_data.rest)].rest, 'rest'] = 1

    # re-order data
    data = data.reset_index().sort_values(['year', 'month'])
    # remove the values for un-needed data
    data.loc[~np.in1d(data.loc[:, 'month'], event_months['rest']), 'rest'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['hot']) & (data.temp > 0)), 'temp'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['cold']) & (data.temp < 0)), 'temp'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['wet']) & (data.precip < 0)), 'precip'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['dry']) & (data.precip > 0)), 'precip'] = 0

    # get previous states.
    for k in ['temp', 'precip', 'rest', 'rest_cum']:
        data.loc[:, 'prev_{}'.format(k)] = data.loc[:, k].shift(1)

    if save:
        for path in save_paths:
            with open(path, 'w') as f:
                f.write('0=normal temp: -1=cold; ; 1=hot; Precip: -1=wet; 1=dry; rest: 1=anomalous restrictions. '
                        'prev_{} = previous months data  \n')
            data.to_csv(path, mode='a', index=False)
    if restrict_cat:
        return data.drop(columns=['rest_cum', 'prev_rest_cum'])
    else:
        data.loc[:, 'rest'] = data.loc[:, 'rest_cum']
        data.loc[:, 'prev_rest'] = data.loc[:, 'prev_rest_cum']
        return data.drop(columns=['rest_cum', 'prev_rest_cum'])


def run_stats():
    print('raw data')
    data = make_data(save=True).dropna()
    out = mutual_info_classif(X=data.drop(columns=['year', 'rest']),
                              y=data.loc[:, 'rest'],
                              discrete_features=True)
    out = pd.Series(out, index=data.drop(columns=['year', 'rest']).columns)
    out.sort_values(inplace=True, ascending=False)
    out.to_csv(os.path.join(outdir, 'mut_info_raw.csv'))

    outdata = pd.DataFrame(index=range(1, 13), columns=data.drop(columns=['year', 'rest', 'month']).columns,
                           dtype=float)
    for m in range(1, 13):
        data = make_data().dropna()
        data = data.loc[data.month == m]
        out = mutual_info_classif(X=data.drop(columns=['year', 'rest', 'month']),
                                  y=data.loc[:, 'rest'],
                                  discrete_features=True)
        cols = data.drop(columns=['year', 'rest', 'month']).columns
        out = pd.Series(out, index=cols)
        outdata.loc[m, cols] = out.loc[cols]

    outdata.to_csv(os.path.join(outdir, 'monthy_mut_info_raw.csv'))

    data = make_data().dropna()
    data.loc[data.temp < 0, 'temp'] = 0
    data.loc[data.precip < 0, 'precip'] = 0
    out = mutual_info_classif(X=data.drop(columns=['year', 'rest']),
                              y=data.loc[:, 'rest'],
                              discrete_features=True)
    out = pd.Series(out, index=data.drop(columns=['year', 'rest']).columns)
    out.sort_values(inplace=True, ascending=False)
    out.to_csv(os.path.join(outdir, 'mut_info_simp.csv'))

    data2 = data.copy(deep=True)
    outdata = pd.DataFrame(index=range(1, 13), columns=data.drop(columns=['year', 'rest', 'month']).columns,
                           dtype=float)
    for m in range(1, 13):
        data = data2.copy(deep=True)
        data = data.loc[data.month == m]
        out = mutual_info_classif(X=data.drop(columns=['year', 'rest', 'month']),
                                  y=data.loc[:, 'rest'],
                                  discrete_features=True)
        cols = data.drop(columns=['year', 'rest', 'month']).columns
        out = pd.Series(out, index=cols)
        outdata.loc[m, cols] = out.loc[cols]

    outdata.to_csv(os.path.join(outdir, 'monthy_mut_info_simp.csv'))


def run_stats2():
    print('raw data')
    data = make_data(save=True,restrict_cat=False).dropna()
    out = mutual_info_regression(X=data.drop(columns=['year', 'rest']),
                              y=data.loc[:, 'rest'])
    out = pd.Series(out, index=data.drop(columns=['year', 'rest']).columns)
    out.sort_values(inplace=True, ascending=False)
    out.to_csv(os.path.join(outdir, 'cum_rest_mut_info_raw.csv'))

    outdata = pd.DataFrame(index=range(1, 13), columns=data.drop(columns=['year', 'rest', 'month']).columns,
                           dtype=float)
    for m in range(1, 13):
        data = make_data(restrict_cat=False).dropna()
        data = data.loc[data.month == m]
        out = mutual_info_regression(X=data.drop(columns=['year', 'rest', 'month']),
                                  y=data.loc[:, 'rest'])
        cols = data.drop(columns=['year', 'rest', 'month']).columns
        out = pd.Series(out, index=cols)
        outdata.loc[m, cols] = out.loc[cols]

    outdata.to_csv(os.path.join(outdir, 'cum_rest_monthy_mut_info_raw.csv'))

    data = make_data(restrict_cat=False).dropna()
    data.loc[data.temp < 0, 'temp'] = 0
    data.loc[data.precip < 0, 'precip'] = 0
    out = mutual_info_regression(X=data.drop(columns=['year', 'rest']),
                              y=data.loc[:, 'rest'])
    out = pd.Series(out, index=data.drop(columns=['year', 'rest']).columns)
    out.sort_values(inplace=True, ascending=False)
    out.to_csv(os.path.join(outdir, 'cum_rest_mut_info_simp.csv'))

    data2 = data.copy(deep=True)
    outdata = pd.DataFrame(index=range(1, 13), columns=data.drop(columns=['year', 'rest', 'month']).columns,
                           dtype=float)
    for m in range(1, 13):
        data = data2.copy(deep=True)
        data = data.loc[data.month == m]
        out = mutual_info_regression(X=data.drop(columns=['year', 'rest', 'month']),
                                  y=data.loc[:, 'rest'])
        cols = data.drop(columns=['year', 'rest', 'month']).columns
        out = pd.Series(out, index=cols)
        outdata.loc[m, cols] = out.loc[cols]

    outdata.to_csv(os.path.join(outdir, 'cum_rest_monthy_mut_info_simp.csv'))


if __name__ == '__main__':

    make_data(True)

    pass
