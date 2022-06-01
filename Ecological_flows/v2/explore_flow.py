"""
created matt_dumont 
on: 2/06/22
"""
import matplotlib.pyplot as plt

from Ecological_flows.v2.alternate_restrictions import naturalise_historical_flow

if __name__ == '__main__':
    #  plot nat vs not nat and the other to see it

    # best guess from the data is 2005-2010... probably 2007
    figsize = (16.5, 9.25)
    data = naturalise_historical_flow()
    # data = data.rolling(10).mean()
    for m in [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]:
        qs = [0.25, 0.5, 0.75]
        fig, axs = plt.subplots(nrows=len(qs), sharex=True, figsize=figsize)
        for ax, q in zip(axs, qs):
            plt_data = data.groupby(['month', 'year']).quantile(q).reset_index()
            temp = plt_data.loc[plt_data.month == m]
            ax.plot(temp.year, temp.flow, c='b', label='record')
            ax.plot(temp.year, temp.nat, c='r', label='nat')
            ax.set_title(f'{q * 100}th percentile')
            ax.legend()
        fig.suptitle(f'month {m}')
        fig.tight_layout()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data.index, data.flow, c='b', label='record')
    ax.plot(data.index, data.nat, c='r', label='nat')
    ax.legend()
    plt.show()
