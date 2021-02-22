"""
 Author: Matt Hanson
 Created: 19/02/2021 1:07 PM
 """
from Storylines.storyline_building_support import make_sampling_options
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_sma_smd_historical, \
    calc_smd_monthly
from Climate_Shocks.get_past_record import get_vcsn_record
from Climate_Shocks import climate_shocks_env
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

figsize = (11, 8)


def compair_means(outdir, detrended=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if detrended:
        raise NotImplementedError
    else:
        vcsn = get_vcsn_record('trended')
        events_path = climate_shocks_env.event_def_path

    vcsn.loc[:, 'tmean'] = (vcsn.loc[:, 'tmax'] + vcsn.loc[:, 'tmin']) / 2
    vcsn.loc[:, 'rain-pet'] = vcsn.loc[:, 'rain'] - vcsn.loc[:, 'pet']
    temp = calc_sma_smd_historical(vcsn.rain, vcsn.pet, vcsn.index, 150, 1)
    vcsn.loc[:, 'smd'] = temp.loc[:, 'smd'].values
    vcsn.loc[:, 'sma'] = temp.loc[:, 'sma'].values
    temp = calc_smd_monthly(vcsn.rain, vcsn.pet, vcsn.index)
    vcsn.loc[:, 'monthly_smd'] = temp
    # todo make monthly basis smd/sma

    vcsn = vcsn.groupby(['year', 'month']).mean()

    events = pd.read_csv(events_path, skiprows=1)
    events = events.set_index(['year', 'month'])
    assert (events.index.values == vcsn.index.values).all()

    # precip
    precip_keys = ['monthly_smd', 'rain', 'rain-pet', 'smd', 'sma', ]
    fig, axs = plt.subplots(len(precip_keys), figsize=figsize)
    fig.suptitle('precip all months')
    for ax, var in zip(axs, precip_keys):
        temp_events = events
        temp_vcsn = vcsn
        data = [
            temp_vcsn.loc[np.isclose(temp_events.loc[:, 'precip'], -1), var].dropna(),
            temp_vcsn.loc[np.isclose(temp_events.loc[:, 'precip'], 0), var].dropna(),
            temp_vcsn.loc[np.isclose(temp_events.loc[:, 'precip'], 1), var].dropna(),
        ]

        ax.boxplot(data, labels=['W', 'A', 'D'])
        ax.set_ylabel(var)
    fig.savefig(os.path.join(outdir, fig._suptitle._text.replace(':', '_') + '.png'))

    for m in range(1, 13):
        fig, axs = plt.subplots(len(precip_keys), figsize=figsize)
        fig.suptitle('precip m:{}'.format(m))
        for ax, var in zip(axs, precip_keys):
            temp_events = events.loc[:, m, :]
            temp_vcsn = vcsn.loc[:, m, :]
            data = [
                temp_vcsn.loc[np.isclose(temp_events.loc[:, 'precip'], -1), var].dropna(),
                temp_vcsn.loc[np.isclose(temp_events.loc[:, 'precip'], 0), var].dropna(),
                temp_vcsn.loc[np.isclose(temp_events.loc[:, 'precip'], 1), var].dropna(),
            ]

            ax.boxplot(data, labels=['W', 'A', 'D'])
            ax.set_ylabel(var)
        fig.savefig(os.path.join(outdir, fig._suptitle._text.replace(':', '_') + '.png'))


    # temp
    for m in range(1, 13):
        fig, axs = plt.subplots(3, figsize=figsize)
        fig.suptitle('temp m:{}'.format(m))
        for ax, var in zip(axs, ['tmin', 'tmax', 'tmean']):
            temp_events = events.loc[:, m, :]
            temp_vcsn = vcsn.loc[:, m, :]
            data = [
                temp_vcsn.loc[np.isclose(temp_events.loc[:, 'temp'], -1), var],
                temp_vcsn.loc[np.isclose(temp_events.loc[:, 'temp'], 0), var],
                temp_vcsn.loc[np.isclose(temp_events.loc[:, 'temp'], 1), var],
            ]

            ax.boxplot(data, labels=['C', 'A', 'H'])
            ax.set_ylabel(var)
        fig.savefig(os.path.join(outdir, fig._suptitle._text.replace(':', '_') + '.png'))
    fig, axs = plt.subplots(3, figsize=figsize)
    fig.suptitle('temp all months')
    for ax, var in zip(axs, ['tmin', 'tmax', 'tmean']):
        temp_events = events
        temp_vcsn = vcsn
        data = [
            temp_vcsn.loc[np.isclose(temp_events.loc[:, 'temp'], -1), var],
            temp_vcsn.loc[np.isclose(temp_events.loc[:, 'temp'], 0), var],
            temp_vcsn.loc[np.isclose(temp_events.loc[:, 'temp'], 1), var],
        ]

        ax.boxplot(data, labels=['C', 'A', 'H'])
        ax.set_ylabel(var)
    fig.savefig(os.path.join(outdir, fig._suptitle._text.replace(':', '_') + '.png'))

    plt.show()


# todo compare to the produced data from the SWG
# todo investigate the delta smd from first to last of month with average start time....
# todo see how the specification shift if given differnt characteristics

if __name__ == '__main__':
    compair_means(r"C:\Users\Matt Hanson\Downloads\compare_means")
