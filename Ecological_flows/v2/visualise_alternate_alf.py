"""
created matt_dumont 
on: 15/06/22
"""
import os.path

import numpy as np

from Ecological_flows.v2.alternate_restrictions import naturalise_historical_flow, make_new_rest_record, new_flows
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import pandas as pd
import project_base

figsize = (11, 8)


def calc_alf(data, key='flow'):
    data = data.copy(deep=True)
    data.loc[:, 'water_year'] = (data.index + pd.DateOffset(months=-6)).year
    data.loc[:, 'alf'] = data.loc[:, key].rolling(7).mean()
    out = data.groupby(['water_year']).min()
    out.loc[:, 'take'] = data.groupby(['water_year']).sum().loc[:, 'take']
    out.reset_index(inplace=True)

    return out


def get_colors(vals, cmap_name='tab10'):
    n_scens = len(vals)
    if n_scens < 20:
        cmap = get_cmap(cmap_name)
        colors = [cmap(e / (n_scens + 1)) for e in range(n_scens)]
    else:
        colors = []
        i = 0
        cmap = get_cmap(cmap_name)
        for v in vals:
            colors.append(cmap(i / 20))
            i += 1
            if i == 20:
                i = 0
    return colors


def malfs():
    data = {}
    nat = naturalise_historical_flow()
    nat.loc[:, 'take'] = 0
    data['naturalised'] = calc_alf(nat, 'nat')

    for nf, v in new_flows.items():
        temp = make_new_rest_record(nf, nat)
        data[nf] = calc_alf(temp)
    keys = [
        'farmer_both',
        'farmer_tail',
        'farmer_front',
        'eco_both',
        'eco_tail',
        'eco_front',
        'current',
        'naturalised',
    ]
    colors = get_colors(keys)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    for c, k in zip(colors, keys):
        v = data[k]
        ax1.plot(v.water_year, v.alf, c=c, label=k + str(new_flows.get(k)))
        ax2.plot(v.water_year, v.loc[:, 'take'], c=c, label=k + ' ' + str(new_flows.get(k)))
    ax1.legend()
    ax1.set_title('Annual Low flows for different restrictions')
    ax2.legend()
    ax2.set_title('Total take for different restrictions')

    fig.tight_layout()
    fig.savefig(os.path.join(project_base.slmmac_dir, "eco_modelling/exloration_figures/historical_malfs.png"))
    # todo plot everything


def varity_of_malfs():
    nyears = [1] + list(range(2, 49, 2))
    data = []
    nat = naturalise_historical_flow()
    alfs = calc_alf(nat, 'nat').alf.values
    num = 10000
    for ny in nyears:
        temp = np.random.choice(alfs, (num, ny))
        data.append(temp.mean(axis=1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, labels=nyears)
    ax.set_xlabel('number of years data to determine MALF')
    ax.set_ylabel('MALF')
    fig.tight_layout()
    fig.savefig(os.path.join(project_base.slmmac_dir, "eco_modelling/exloration_figures/resample_malfs.png"))


if __name__ == '__main__':
    varity_of_malfs()
    malfs()
