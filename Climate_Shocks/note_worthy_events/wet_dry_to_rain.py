"""
 Author: Matt Hanson
 Created: 22/12/2020 9:41 AM
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from Climate_Shocks.note_worthy_events.final_event_recurance import backed_dir, get_org_data
from Climate_Shocks.get_past_record import get_vcsn_record


def make_cum_rain_day_rain():
    weather = get_vcsn_record()
    weather.loc[:, 'ndays_rain'] = (weather.loc[:, 'rain'] > 0.01).astype(float)
    weather = weather.groupby(['month', 'year']).sum().loc[:, ['rain', 'ndays_rain']]
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
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    for ax, k in zip((ax1,ax2),['rain', 'ndays_rain']):
        ax.set_title(k)
        patches = []
        for i, (df, nm, c) in enumerate(zip([wet, normal, dry], ['wet', 'normal', 'dry'],['r','g','b'])):
            plt_data =[]
            for m in range(1,13):
                plt_data.append(df.loc[df.month==m, k])

            lbls = range(1,13)
            bplot = ax.boxplot(plt_data, labels=lbls, positions=np.arange(1,13)*3+i, patch_artist=True,)

            for patch in bplot['boxes']:
                patch.set_facecolor(c)
            patches.append(mpatches.Patch(color=c, label=nm))
        # set legend
        ax.legend(handles=patches)
    plt.show()

if __name__ == '__main__':
    plt_data()
