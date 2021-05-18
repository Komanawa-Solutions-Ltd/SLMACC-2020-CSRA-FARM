"""
 Author: Matt Hanson
 Created: 18/05/2021 12:08 PM
 """
import ksl_env
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Pasture_Growth_Modelling.historical_average_baseline import get_historical_average_baseline, \
    run_past_basgra_dryland


def try_bias_correction():  # todo check on this, does this 'fix' our problems
    outdir = os.path.join(ksl_env.slmmac_dir,'outputs_for_ws', 'norm','historical_trended v historical plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    historical = run_past_basgra_dryland(site='oxford')
    historical.loc[:, 'year'] = historical.loc[:, 'year'].astype(int)
    historical = historical.groupby(['year', 'month']).mean()

    quantified = pd.read_csv(os.path.join(
        ksl_env.slmmac_dir,
        r"outputs_for_ws\norm\historical_quantified_1yr_trend\oxford-dryland\m_PGR.csv"),
        skiprows=1
    )
    quantified.loc[:, 'year'] = quantified.loc[:, 'month'].str.split('-').str[1].astype(int)
    quantified.set_index('year', inplace=True)

    use_data = pd.DataFrame(index=historical.index)
    for y, m in historical.index.values:
        use_data.loc[(y, m), 'historical'] = historical.loc[(y, m), 'pg']
        try:
            use_data.loc[(y, m), 'quantified'] = quantified.loc[y, str(m)]
        except KeyError:
            pass
    use_data = use_data.reset_index()
    temp = use_data.groupby('month').mean()
    temp2 = (temp.loc[:, 'historical'] / temp.loc[:, 'quantified']).to_dict()
    use_data.loc[:, 'quantified_corr'] = use_data.loc[:, 'quantified'] * [temp2[e] for e in use_data.month]
    pass
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(16.8, 9.53))
    axs = axs.flatten()
    for m, ax in zip(range(1, 13),axs):
        versions = ['historical', 'quantified', 'quantified_corr']
        ax.boxplot([use_data.loc[use_data.month == m, e].dropna() for e in versions],
                   labels=versions)
        ax.set_title(m)
        pass
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'time_series.png'))
    baseline = \
    pd.read_csv(os.path.join(ksl_env.slmmac_dir, r"outputs_for_ws\norm\Baseline\oxford-dryland\m_PGR.csv"),
                skiprows=1).loc[0].iloc[1:13]
    baseline.index = baseline.index.values.astype(int)
    plt_data = use_data.groupby('month').mean().drop(columns='year')
    idx = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    fig, ax = plt.subplots(figsize=(16.8, 9.53))
    ax.plot(range(12), plt_data.loc[idx, 'historical'], c='r', label='historical')
    ax.plot(range(12), plt_data.loc[idx, 'quantified'], c='b', label='quantified')
    ax.plot(range(12), plt_data.loc[idx, 'quantified_corr'], c='y', label='quantified_corr')
    ax.plot(range(12), baseline.loc[idx], c='g', label='baseline raw')
    ax.plot(range(12), baseline.loc[idx] * [temp2[e] for e in idx], c='k', label='baseline corrected')
    ax.set_xticklabels(idx)
    ax.legend()
    fig.savefig(os.path.join(outdir,'boxplots.png'))




if __name__ == '__main__':
    try_bias_correction()
    # todo show this to zeb