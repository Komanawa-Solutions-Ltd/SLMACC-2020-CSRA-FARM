"""
 Author: Matt Hanson
 Created: 22/12/2020 9:41 AM
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import os
from Climate_Shocks.note_worthy_events.final_event_recurance import get_org_data
from Climate_Shocks.climate_shocks_env import event_def_dir as backed_dir
from Climate_Shocks.note_worthy_events.simple_soil_moisture import calc_sma_smd
from Climate_Shocks.get_past_record import get_vcsn_record


def make_cum_rain_day_rain():
    weather = get_vcsn_record()
    weather.loc[:, 'ndays_rain'] = (weather.loc[:, 'rain'] > 0.01).astype(float)
    weather.loc[:, 'tmean'] = (weather.loc[:, 'tmax'] + weather.loc[:, 'tmax']) / 2
    weather = weather.groupby(['month', 'year']).agg({'rain': 'sum', 'ndays_rain': 'sum', 'tmean': 'mean'})
    temp_data = get_org_data()
    data = pd.DataFrame(index=pd.MultiIndex.from_product([range(1, 13), range(1972, 2020)], names=['month', 'year']),
                        columns=['temp', 'precip'], dtype=float)

    data.loc[:, :] = 0
    data.loc[temp_data.loc[pd.notna(temp_data.hot)].hot, 'temp'] = 1
    data.loc[temp_data.loc[pd.notna(temp_data.cold)].cold, 'temp'] = -1
    data.loc[temp_data.loc[pd.notna(temp_data.dry)].dry, 'precip'] = 1
    data.loc[temp_data.loc[pd.notna(temp_data.wet)].wet, 'precip'] = -1

    outdata = pd.merge(weather, data, left_index=True, right_index=True)
    return outdata.reset_index()


def comp_data():
    data = make_cum_rain_day_rain()
    wet = data.loc[data.precip < -0.1]
    normal = data.loc[np.isclose(data.precip, 0)]
    dry = data.loc[data.precip > 0.1]

    for df, nm in zip([wet, normal, dry], ['wet', 'normal', 'dry']):
        df.groupby('month').describe().loc[:, ['rain',
                                               'ndays_rain']].to_csv(os.path.join(backed_dir,
                                                                                  'rain_based_{}.csv'.format(nm)))


def plt_data():
    data = make_cum_rain_day_rain()
    wet = data.loc[data.precip < -0.1]
    normal = data.loc[np.isclose(data.precip, 0)]
    dry = data.loc[data.precip > 0.1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    for ax, k in zip((ax1, ax2, ax3), ['rain', 'ndays_rain', 'tmean']):
        ax.set_title(k)
        patches = []
        for i, (df, nm, c) in enumerate(zip([wet, normal, dry], ['wet', 'normal', 'dry'], ['r', 'g', 'b'])):
            plt_data = []
            for m in range(1, 13):
                plt_data.append(df.loc[df.month == m, k])

            lbls = range(1, 13)
            bplot = ax.boxplot(plt_data, labels=lbls, positions=np.arange(1, 13) * 3 + i, patch_artist=True, )

            for patch in bplot['boxes']:
                patch.set_facecolor(c)
            patches.append(mpatches.Patch(color=c, label=nm))
        # set legend
        ax.legend(handles=patches)


def plt_data2():
    data = make_cum_rain_day_rain()
    wet = data.loc[data.precip < -0.1]
    normal = data.loc[np.isclose(data.precip, 0)]
    dry = data.loc[data.precip > 0.1]

    for m, (k1, k2) in itertools.product([4, 5, 8, 9], zip(['rain', 'ndays_rain'], ['tmean', 'tmean'])):
        fig, ax = plt.subplots()
        ax.set_title('month{}: {}-{}'.format(m, k1, k2))
        ax.set_ylabel(k2)
        ax.set_xlabel(k1)
        for i, (df, nm, c) in enumerate(zip([wet, normal, dry], ['wet', 'normal', 'dry'], ['r', 'g', 'b'])):
            idx = df.month == m
            ax.scatter(df.loc[idx, k1], df.loc[idx, k2], c=c, label=nm)
        ax.legend()


def plt_data3():
    data = get_vcsn_record().reset_index()
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2

    temp = calc_sma_smd(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]
    data.loc[:, 'roll_rain_5'] = data.loc[:, 'rain'].rolling(5).sum()
    data.loc[:, 'roll_tmean_5'] = data.loc[:, 'tmean'].rolling(5).mean()
    data.loc[:, 'roll_rain_10'] = data.loc[:, 'rain'].rolling(10).sum()
    data.loc[:, 'roll_rain_15'] = data.loc[:, 'rain'].rolling(15).sum()
    data.loc[:, 'roll_rain_20'] = data.loc[:, 'rain'].rolling(20).sum()

    for m in range(1,13):
        fig, axs = plt.subplots(ncols=3, sharey=True)
        fig.suptitle('month:{}'.format(m))
        idx = data.month == m
        for ax, v in zip(axs.flatten(), ['roll_rain_10', 'roll_rain_15', 'roll_rain_20']):
            ax.scatter(data.loc[idx, v], data.loc[idx, 'sma'])
            ax.set_xlabel(v)


def plt_data4():
    data = get_vcsn_record().reset_index()
    data.loc[:, 'tmean'] = (data.loc[:, 'tmax'] + data.loc[:, 'tmin']) / 2

    temp = calc_sma_smd(data['rain'], data['pet'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:, trans_cols] = temp.loc[:, trans_cols]
    data.loc[:, 'roll_rain_5'] = data.loc[:, 'rain'].rolling(5).sum()
    data.loc[:, 'roll_tmean_5'] = data.loc[:, 'tmean'].rolling(5).mean()
    data.loc[:, 'roll_rain_10'] = data.loc[:, 'rain'].rolling(10).sum()
    data.loc[:, 'roll_rain_30'] = data.loc[:, 'rain'].rolling(30).sum()

    for m in [4,5,8,9]:
        idx = data.month == m
        fig, ax = plt.subplots()
        t = ax.scatter(data.loc[idx, 'roll_rain_5'], data.loc[idx, 'roll_tmean_5'], c=data.loc[idx, 'sma'])
        fig.colorbar(t)
        fig.suptitle('month:{}'.format(m))
        ax.set_xlabel('roll_rain_5')
        ax.set_ylabel('roll_tmean_5')

        fig, ax = plt.subplots()
        t = ax.scatter(data.loc[idx, 'roll_rain_10'], data.loc[idx, 'roll_tmean_5'], c=data.loc[idx, 'sma'])
        fig.colorbar(t)
        fig.suptitle('month:{}'.format(m))
        ax.set_xlabel('roll_rain_10')
        ax.set_ylabel('roll_tmean_5')



if __name__ == '__main__':
    # plt_data()
    # plt_data2()
    plt_data3()
    plt.show()
