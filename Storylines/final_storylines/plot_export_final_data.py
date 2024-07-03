"""
created matt_dumont 
on: 12/07/22
"""
import itertools
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt
import pandas as pd

import project_base
from Storylines.storyline_building_support import month_len
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite


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
    cor_data = get_1yr_data(correct=True)
    raw_data = get_1yr_data(correct=False)
    for nm, correct in itertools.product(['baseline', 'scare', 'hurt'], [True, False]):
        if correct:
            corn = 'mod'
            use_data = cor_data.copy()
        else:
            corn = 'raw'
            use_data = raw_data.copy()
        use_data.loc[:, 'k'] = use_data.loc[:, 'ID'] + '_' + use_data.loc[:, 'irr_type'] + '_irr'
        use_data.set_index('k', inplace=True)

        slids = get_storyline_ids(nm)
        out[f'{nm}-{corn}'] = use_data.loc[slids]
    return out


def export_data():
    outir = Path(__file__).parent.joinpath('scenario_pg_data')
    outir.mkdir(exist_ok=True)
    data = get_scenario_data()
    for k, v in data.items():
        v.to_csv(outir.joinpath(f'{k}.csv'))


def export_data_summary():
    outdir = Path().home().joinpath('Downloads/scen_data')
    outdir.mkdir(exist_ok=True)
    data = get_scenario_data()
    for k, temp in data.items():
        keys = pd.Series(temp.keys())
        keys = keys.loc[keys.str.contains('yr1')]
        temp = temp.loc[:, keys].mean()
        temp = (temp / 1000).round(2).sort_index()
        temp.to_csv(outdir.joinpath(f'{k}.csv'))


def print_baseline_limits():
    outdir = Path().home().joinpath('Downloads/scen_data')
    outdir.mkdir(exist_ok=True)
    data = get_scenario_data()['baseline-raw']
    keys = pd.Series(data.keys())
    keys = keys.loc[keys.str.contains('yr1')]
    data = (data.loc[:, keys] / 1000).round(2)
    data.describe().transpose().sort_index().to_csv(outdir.joinpath('baseline_limits.csv'))


def plot_storage():
    outdir = Path(project_base.slmmac_dir).joinpath('0_Y2_and_Final_Reporting', 'final_plots', 'storage_scens')
    outdir.mkdir(exist_ok=True)
    scen_data = get_scenario_data()
    scens = [
        'baseline-raw',
        'scare-raw',
        'hurt-raw'
    ]
    sites = ['eyrewell', 'oxford']
    modes = ['irrigated',
             'store400',
             'store600',
             'store800',
             ]
    colors = {
        'irrigated': 'b',
        'store400': 'm',
        'store600': 'y',
        'store800': 'c',
    }
    for site in sites:
        fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(8, 10))
        for i, (scen, ax) in enumerate(zip(scens, axs)):
            for mode in modes:
                c = colors[mode]
                plot_months, pdata = _prep_data(scen_data[scen], f'{site}-{mode}', True)
                ax.plot(range(12), pdata, label=mode, c=c)
            ax.set_title(scen.strip('-raw'))
            ax.set_ylim(1, 100)
            if i == 2:
                ax.legend()
                ax.set_xticks(range(0, 12))
                ax.set_xticklabels([month_to_month[m] for m in plot_months])
            pass
        fig.suptitle(site.capitalize())
        fig.supylabel('Pasture Growth kg DM/ha/day')
        fig.tight_layout()
        plt.show()
        fig.savefig(outdir.joinpath(f'{site}_storage.png'))


def plot_base():
    outdir = Path(project_base.slmmac_dir).joinpath('0_Y2_and_Final_Reporting', 'final_plots', 'base_scens')
    outdir.mkdir(exist_ok=True)
    sms = [
        'eyrewell-irrigated',
        'oxford-irrigated',
        'oxford-dryland',
    ]
    scen_data = get_scenario_data()
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(8, 10))
    for sm, ax in zip(sms, axs):
        pass
        # make datasets
        raw_data = get_nyr_suite(1, sm.split('-')[0], sm.split('-')[1], monthly_data=True)
        plot_months, pdata = _prep_data(raw_data, sm, resampled=True)
        vparts = ax.violinplot(pdata, widths=0.5, showextrema=False)
        for p in vparts['bodies']:
            p.set_color('grey')
        scens = [
            'baseline-raw',
            'scare-raw',
            'hurt-raw'
        ]
        colors = [
            'b',
            'orange',
            'red',
        ]
        for sen, c in zip(scens, colors):
            plot_months, pdata = _prep_data(scen_data[sen], sm, True)
            ax.plot(range(1, 13), pdata, label=sen.strip('-raw'), c=c)
        if sm == 'oxford-dryland':
            ax.legend()
        ax.set_title(sm.capitalize())
        ax.set_ylim(0, 100)
    fig.supylabel('Pasture Growth kg DM/ha/day')
    fig.tight_layout()
    fig.savefig(outdir.joinpath('base_scenarios.png'))


def _prep_data(data, sm, take_mean=False, resampled=False):
    plot_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    outdata = []
    extra = ''
    if resampled:
        extra = 'yr0_'
    for m in plot_months:
        if take_mean:
            outdata.append(data.loc[:, f'{sm}_pg_{extra}m{m:02d}'].dropna().values.mean() / month_len[m])
        else:
            outdata.append(data.loc[:, f'{sm}_pg_{extra}m{m:02d}'].dropna().values / month_len[m])

    return plot_months, outdata


month_to_month = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec',

}
if __name__ == '__main__':
    plot_base()
