"""
 Author: Matt Hanson
 Created: 21/12/2020 8:47 AM
 """
import pandas as pd
import numpy as np
import os
import ksl_env
from sklearn.feature_selection import mutual_info_classif
from Climate_Shocks.note_worthy_events.final_event_recurance import get_org_data, backed_dir

outdir = ksl_env.shared_drives(r"SLMACC_2020\event_definition\mutual_info")
if not os.path.exists(outdir):
    os.makedirs(outdir)

event_months = {
    'hot': [11, 12, 1, 2, 3],
    'cold': list(range(5, 10)),
    'dry': list(range(8, 13)) + list(range(1, 6)),
    'wet': list(range(5, 10)),
    'rest': list(range(9, 13)) + list(range(1, 5))}


def make_data(save=False):
    temp_data = get_org_data()
    data = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 13), range(1972, 2020)], names=['month', 'year']),
                        columns=['temp', 'precip', 'rest'], dtype=int)
    data.loc[:, :] = 0
    data.loc[temp_data.loc[pd.notna(temp_data.hot)].hot, 'temp'] = 1
    data.loc[temp_data.loc[pd.notna(temp_data.cold)].cold, 'temp'] = -1
    data.loc[temp_data.loc[pd.notna(temp_data.dry)].dry, 'precip'] = 1
    data.loc[temp_data.loc[pd.notna(temp_data.wet)].wet, 'precip'] = -1
    data.loc[temp_data.loc[pd.notna(temp_data.rest)].rest, 'rest'] = 1

    # re-order data
    data = data.reset_index().sort_values(['year', 'month'])
    # remove the values for un-needed data
    data.loc[~np.in1d(data.loc[:, 'month'], event_months['rest']), 'rest'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['hot']) & (data.temp > 0)), 'temp'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['cold']) & (data.temp < 0)), 'temp'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['wet']) & (data.precip < 0)), 'precip'] = 0
    data.loc[(~np.in1d(data.loc[:, 'month'], event_months['dry']) & (data.precip > 0)), 'precip'] = 0

    # get previous states.
    for k in ['temp', 'precip', 'rest']:
        data.loc[:, 'prev_{}'.format(k)] = data.loc[:, k].shift(1)

    if save:
        path = os.path.join(backed_dir, 'annual_event_data.csv')
        with open(path,'w') as f:
            f.write('0=normal temp: -1=cold; ; 1=hot; Precip: -1=wet; 1=dry; rest: 1=anomalous restrictions. '
                    'prev_{} = previous months data  \n')
        data.to_csv(path,mode='a')
    return data


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
        data = data.loc[data.month==m]
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
        data = data.loc[data.month==m]
        out = mutual_info_classif(X=data.drop(columns=['year', 'rest', 'month']),
                                  y=data.loc[:, 'rest'],
                                  discrete_features=True)
        cols = data.drop(columns=['year', 'rest', 'month']).columns
        out = pd.Series(out, index=cols)
        outdata.loc[m, cols] = out.loc[cols]

    outdata.to_csv(os.path.join(outdir, 'monthy_mut_info_simp.csv'),index=False)



if __name__ == '__main__':
    data = make_data()
    pass
