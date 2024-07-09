"""
created matt_dumont 
on: 7/9/24
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import project_base
from komanawa.slmacc_csra import get_historical_pg_data, get_historical_quantified_data, get_1yr_non_exceedence_prob, \
    get_nyr_non_exceedence_prob

colors = {
    'Historical': 'orangered',
    'Historical quantized trended': 'red',
    'Historical quantized de-trended': 'royalblue',
    'Random suite': 'purple',
}

savedir = project_base.slmmac_dir.joinpath('0_Y2_and_Final_Reporting', 'final_plots', 'historical_vs_suite')
savedir.mkdir(exist_ok=True)


def plot_historical_vs_1yr():
    modes = ['irrigated', 'dryland']
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(10, 8))
    fig.suptitle('Historical vs 1yr non-exceedance probability')
    fig.supxlabel('Pasture Growth (tons DM / ha / year)')
    fig.supylabel('Cumulative Probability (%)')
    j = 0
    for i, (site, mode) in enumerate(itertools.product(['eyrewell', 'oxford'], modes)):
        if site == 'eyrewell' and mode == 'dryland':
            continue
        ax = axs[j]
        j += 1
        ax.set_title(f'{site.capitalize()}-{mode.capitalize()}')

        hist = get_historical_pg_data(site, mode)
        hist = hist.groupby(['year']).sum()['pg']
        hist = _make_percentiles(hist.values)
        ax.plot(hist.index / 1000, hist.values, label='Historical', color=colors['Historical'])

        for detrended, lab in zip([False, True], ['Historical quantized trended', 'Historical quantized de-trended']):
            hist = get_historical_quantified_data(site, mode, detrended)
            hist = _make_percentiles(hist['annual_pg'].values)
            ax.plot(hist.index / 1000, hist.values, label=lab, color=colors[lab])

        prob = pd.DataFrame(get_1yr_non_exceedence_prob(site, mode, None)).reset_index()
        prob = prob.sort_values('pg')
        ax.plot(prob['pg'] / 1000, prob['prob'], label='Random suite', color=colors['Random suite'])
        ax.legend()

    # legend...
    fig.tight_layout()
    fig.savefig(savedir.joinpath('historical_vs_1yr.png'))
    plt.close(fig)


def _make_percentiles(x):
    pers = np.arange(0, 101)
    out = pd.Series(index=np.nanpercentile(x, pers), data=pers)
    return out


def _make_percentiles_nyr(x, nyr):
    pers = np.arange(0, 101)
    usedata = grouper(nyr, x, np.nan)
    usedata = np.sum(usedata, axis=1)
    out = pd.Series(index=np.nanpercentile(usedata, pers), data=pers)
    return out


def plot_historical_vs_suite_nyear(nyr):
    modes = ['irrigated', 'dryland']
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(10, 8))
    fig.suptitle(f'Historical vs {nyr}yr non-exceedance probability')
    fig.supxlabel(f'Pasture Growth (tons DM / ha / {nyr} year)')
    fig.supylabel('Cumulative Probability (%)')
    j = 0
    for i, (site, mode) in enumerate(itertools.product(['eyrewell', 'oxford'], modes)):
        if site == 'eyrewell' and mode == 'dryland':
            continue
        ax = axs[j]
        j += 1
        ax.set_title(f'{site.capitalize()}-{mode.capitalize()}')

        prob = pd.DataFrame(get_nyr_non_exceedence_prob(nyr, site, mode, None)).reset_index()
        prob = prob.sort_values('pg')
        ax.plot(prob['pg'] / 1000, prob['prob'], label='Random suite', color=colors['Random suite'])

        minpg = prob['pg'].min() / 1000
        maxpg = prob['pg'].max() / 1000

        hist = get_historical_pg_data(site, mode)
        hist = hist.groupby(['year']).sum()['pg']
        hist = _make_percentiles_nyr(hist.values, nyr)
        x = np.concatenate(([minpg], hist.index / 1000, [maxpg]))
        y = np.concatenate(([0], hist.values, [100]))
        ax.plot(x, y, label='Historical', color=colors['Historical'])

        for detrended, lab in zip([False, True], ['Historical quantized trended', 'Historical quantized de-trended']):
            hist = get_historical_quantified_data(site, mode, detrended)
            hist = _make_percentiles_nyr(hist['annual_pg'].values, nyr)
            x = np.concatenate(([minpg], hist.index / 1000, [maxpg]))
            y = np.concatenate(([0], hist.values, [100]))
            ax.plot(x, y, label=lab, color=colors[lab])

        ax.legend()

    fig.tight_layout()
    fig.savefig(savedir.joinpath(f'historical_vs_{nyr}yr.png'))
    plt.close(fig)


def grouper(n, iterable, fillvalue=None):
    """
    group an iterable into n sized chunks

    :param n: number of elements in each chunk
    :param iterable: the iterable to chunk
    :param fillvalue: the value to fill the last chunk with if it is not full
    :return: list of n sized chunks
    """
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    out = []
    temp = itertools.zip_longest(fillvalue=fillvalue, *args)
    for t in temp:
        mini_out = []
        for l in t:
            if l is not None:
                mini_out.append(l)
        out.append(np.array(mini_out))

    return out


if __name__ == '__main__':
    plot_historical_vs_1yr()
    for n in [2, 3, 5, 10]:
        plot_historical_vs_suite_nyear(n)
