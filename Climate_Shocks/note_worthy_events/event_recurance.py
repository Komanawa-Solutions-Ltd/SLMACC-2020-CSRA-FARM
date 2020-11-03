"""
 Author: Matt Hanson
 Created: 3/11/2020 9:04 AM
 """

from Climate_Shocks.note_worthy_events.vcsn_pull import vcsn_pull_single_site
from Climate_Shocks.note_worthy_events.simple_smd_soilt import calc_smd, calc_sma_smd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_vcsn_smd():
    data, use_cords = vcsn_pull_single_site(r"D:\VCSN",
                                            lat=-43.358,
                                            lon=172.301,
                                            year_min=1972,
                                            year_max=2019,
                                            use_vars=('evspsblpot', 'pradj', 'pr'))
    print(use_cords)

    temp = calc_sma_smd(data['pr'], data['evspsblpot'], data.date, 150, 1)

    trans_cols = ['mean_doy_smd', 'sma', 'smd', 'drain', 'aet_out']
    data.loc[:,trans_cols] = temp.loc[:,trans_cols]

    data.set_index('date', inplace=True)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.plot(data.index, data['evspsblpot'], label='pet')
    ax1.plot(data.index, data['aet_out'], label='aet')
    ax2.plot(data.index, data['pr'], label='rain')
    ax3.plot(data.index, data['smd'], label='smd')
    ax3.plot(data.index, data['mean_doy_smd'], label='daily_mean_smd')
    ax4.plot(data.index, data['sma'], label='sma')
    ax4.axhline(ls='--', c='k')

    for ax in (ax1, ax2, ax3, ax4):
        ax.legend()

    plt.show()

def check_vcns_data():
    data, use_cords = vcsn_pull_single_site(r"D:\VCSN",
                                            lat=-43.358,
                                            lon=172.301,
                                            year_min=1972,
                                            year_max=2019,
                                            use_vars='all')

    print(use_cords)
    data.set_index('date', inplace=True)
    for v in data.keys():
        fix,(ax) = plt.subplots()
        ax.plot(data.index, data[v])
        ax.set_title(v)
    plt.show()


if __name__ == '__main__':
    plot_vcsn_smd()
