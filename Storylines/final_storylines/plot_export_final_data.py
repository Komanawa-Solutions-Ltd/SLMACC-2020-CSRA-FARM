"""
created matt_dumont 
on: 12/07/22
"""
import itertools
from pathlib import Path
import zipfile

import pandas as pd

from Storylines.storyline_runs.run_random_suite import get_1yr_data

def get_storyline_ids(nm):
    base_path = Path(__file__).parent
    if nm == 'baseline':
        mod = ''
        fn = 'Final_most_probable.zip'
    elif nm == 'scare':
        mod = '_irr'
        fn = 'Final_scare_autumn_drought2_mon_thresh_all.zip'
    elif nm == 'hurt':
        mod = '_irr'
        fn = 'Final_hurt_hurt_v1_storylines_cluster_004.zip'
    else:
        raise ValueError('unexpected value')

    with zipfile.ZipFile(base_path.joinpath(fn)) as f:
        file_list = f.namelist()
        print(file_list[0])
        file_list = file_list[1:]
    names = [Path(e).name.strip('.csv').strip() + mod for e in file_list]

    return names

def get_scenario_data():
    out = {}
    cor_data =  get_1yr_data(correct=True)
    raw_data =  get_1yr_data(correct=False)
    for nm, correct in itertools.product(['baseline','scare','hurt'], [True, False]):
        if correct:
            corn = 'mod'
            use_data = cor_data.copy()
        else:
            corn = 'raw'
            use_data = raw_data.copy()
        use_data.loc[:,'k'] = use_data.loc[:,'ID'] + '_' + use_data.loc[:,'irr_type'] + '_irr'
        use_data.set_index('k', inplace=True)

        slids = get_storyline_ids(nm)
        out[f'{nm}-{corn}'] = use_data.loc[slids]
    return out

def export_data():
    outir = Path(__file__).parent.joinpath('scenario_pg_data')
    outir.mkdir(exist_ok=True)
    data = get_scenario_data()
    for k,v in data.items():
        v.to_csv(outir.joinpath(f'{k}.csv'))

def export_data_summary():
    outdir = Path().home().joinpath('Downloads/scen_data')
    outdir.mkdir(exist_ok=True)
    data = get_scenario_data()
    for k,temp in data.items():
        keys = pd.Series(temp.keys())
        keys = keys.loc[keys.str.contains('yr1')]
        temp = temp.loc[:,keys].mean()
        temp = (temp/1000).round(2).sort_index()
        temp.to_csv(outdir.joinpath(f'{k}.csv'))

def print_baseline_limits():
    outdir = Path().home().joinpath('Downloads/scen_data')
    outdir.mkdir(exist_ok=True)
    data = get_scenario_data()['baseline-raw']
    keys = pd.Series(data.keys())
    keys = keys.loc[keys.str.contains('yr1')]
    data = (data.loc[:,keys]/1000).round(2)
    data.describe().transpose().sort_index().to_csv(outdir.joinpath('baseline_limits.csv'))

def plot_storage(): # todo see hydrosoc poster
    raise NotImplementedError

def plot_base(): # todo
    sms = ['oxford-dryland', 'oxford-irrigated', 'eyrewell-irrigated']
    raise NotImplementedError

if __name__ == '__main__':
    print_baseline_limits()
