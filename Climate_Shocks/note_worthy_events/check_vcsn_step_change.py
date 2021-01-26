"""
 Author: Matt Hanson
 Created: 4/12/2020 9:40 AM
 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from Climate_Shocks.get_past_record import get_vcsn_record, get_restriction_record, sites

weather_keys = [
    'pet',
    'radn',
    'tmean',
    'f_rest',
    'rain',
]

agg_dict = {'pet': 'sum', 'radn': 'sum', 'tmax': 'mean', 'tmin': 'mean', 'rain': 'sum', 'f_rest': 'sum', 'date':'first'}


def plot_vcsn_for_step_change(data, rolling_window, **kwargs):
    fig, axs = plt.subplots(len(weather_keys))
    cmap = get_cmap('tab20')
    n_scens = len(data.keys())
    colors = [cmap(e / n_scens) for e in range(n_scens)]  # pick from color map
    for k, ax in zip(weather_keys, axs):
        for (dk, d), c in zip(data.items(), colors):
            ax.plot(d['date'], d[k], label='{}_{}'.format(dk, k), c=c,**kwargs)
            t = d.rolling(rolling_window).agg('mean')
            t.loc[:,'date'] = d.loc[:,'date']
            ax.plot(t['date'], t[k], c=c,**kwargs)


    for ax in axs:
        ax.legend()
    plt.show()

def make_annual_vcsn_res_data(site, agg_per):
    weather_data = get_vcsn_record(site=site)
    rest_data = get_restriction_record().loc[:, ['f_rest']]
    annual_data = pd.merge(weather_data, rest_data, right_index=True, left_index=True).reset_index()
    annual_data = annual_data.groupby(agg_per).agg(agg_dict)
    annual_data.loc[:,'tmean'] = (annual_data.loc[:,'tmax'] + annual_data.loc[:,'tmin'])/2
    return annual_data

if __name__ == '__main__':
    agg_per = ['year', 'month']
    data = {e: make_annual_vcsn_res_data(e, agg_per) for e in sites}
    plot_vcsn_for_step_change(data, 24)
    # note I am content that there is not any terrible step change

