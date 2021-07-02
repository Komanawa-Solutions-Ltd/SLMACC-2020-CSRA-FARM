"""
 Author: Matt Hanson
 Created: 2/07/2021 8:49 AM
 """
import ksl_env
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp, default_pasture_growth_dir, pgm_log_dir, \
    default_mode_sites
from Storylines.storyline_evaluation.storyline_eval_support import calc_cumulative_impact_prob
import matplotlib.pyplot as plt
import os
import gc
import numpy as np
import pandas as pd
from scipy.stats import kde, binned_statistic_2d
import itertools
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite
from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched, calc_impact_prob, \
    calc_cumulative_impact_prob
import time
from copy import deepcopy

figsize = (16.5, 9.25)
base_color = 'limegreen'
base_ls = 'dashdot'
base_lw = 2
base_outdir = os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', 'cumulative_historical_trend')
if not os.path.exists(base_outdir):
    os.makedirs(base_outdir)


def yr1_cumulative_probability():
    outdir = os.path.join(base_outdir, '1yr')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data = pd.read_csv(os.path.join(
        ksl_env.mh_unbacked("SLMACC_2020_norm\pasture_growth_sims\historical_quantified_1yr_trend\IID_probs_pg.csv")))

    for mode, site in default_mode_sites:
        print(site, mode)
        pgr = data.loc[:, f'{site}-{mode}_pg'].dropna().values / 1000
        prob = np.zeros(pgr.shape) + 1
        _plot_cum(pgr=pgr, prob=prob, step_size=0.1, site=site, mode=mode, nyr=1, outdir=outdir)


def nyr_cumulative_prob(nyr):
    outdir = os.path.join(base_outdir, f'{nyr}yr')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data = pd.read_csv(os.path.join(
        ksl_env.mh_unbacked("SLMACC_2020_norm\pasture_growth_sims\historical_quantified_1yr_trend\IID_probs_pg.csv")))
    data = data.rolling(nyr).sum()
    for mode, site in default_mode_sites:
        pgr = data.loc[:, f'{site}-{mode}_pg'].dropna().values / 1000
        prob = np.zeros(pgr.shape) + 1
        fig = _plot_cum(pgr=pgr, prob=prob, step_size=0.1, site=site, mode=mode, nyr=nyr, outdir=outdir)


def _plot_cum(pgr, prob, step_size, site, mode, nyr, outdir):
    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=pgr,
                                                    prob=prob, stepsize=step_size)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
    ax1.bar(cum_pgr, cum_prob, width=step_size / 2, color='grey')
    ax1.set_title('Exceedance probability')
    ax1.set_xlabel('Pasture growth tons DM/Ha/year')
    ax1.axvline(get_pgr_prob_baseline_stiched(nyr, site, mode) / 1000, c=base_color, lw=base_lw, ls=base_ls,
                label='baseline pasture growth')
    ax1.set_ylabel('Probability of an event with \nequal or greater Pasture growth')
    ax1.legend()

    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=pgr,
                                                    prob=prob, stepsize=step_size,
                                                    more_production_than=False)
    temp = pd.DataFrame({'prob': cum_prob, 'pg': cum_pgr})
    temp.to_csv(os.path.join(outdir, f'{site}-{mode}_{nyr}yr_cumulative_exceed_prob.csv'))

    ax2.bar(cum_pgr, cum_prob, width=step_size / 2, color='grey')
    ax2.set_title('Non-exceedance probability')
    ax2.set_xlabel('Pasture growth anomaly tons DM/Ha/year')
    ax2.axvline(get_pgr_prob_baseline_stiched(nyr, site, mode) / 1000,
                c=base_color, lw=base_lw, ls=base_ls, label='baseline pasture growth')
    ax2.set_ylabel('Probability of an event with \nequal or less Pasture growth')
    ax2.legend()

    nm = f'{site}-{mode}_exceed_prob'
    fig.suptitle(f'{site}-{mode} {nyr} year from historical data (non-stationary)'.capitalize())
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{site}-{mode}_{nyr}yr_cumulative_exceed_prob.png'))
    return fig


if __name__ == '__main__':
    yr1_cumulative_probability()
    for y in [2, 3, 5, 10]:
        print(y)
        nyr_cumulative_prob(y)
