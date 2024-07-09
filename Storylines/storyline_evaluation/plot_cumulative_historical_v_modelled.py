"""
 Author: Matt Hanson
 Created: 2/07/2021 11:50 AM
 """
import project_base
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Pasture_Growth_Modelling.basgra_parameter_sets import default_mode_sites
from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched, calc_impact_prob, \
    calc_cumulative_impact_prob

figsize = (16.5, 9.25)
base_color = 'limegreen'
base_ls = 'dashdot'
base_lw = 2
base_trend = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'norm', 'cumulative_historical_trend')
base_detrend = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'norm', 'cumulative_historical_detrend')
base_model = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'norm', 'random_scen_plots')

# data developed by Storylines/storyline_evaluation/plot_historical_detrended.py and
# Storylines/storyline_evaluation/plot_historical_trended.py
import warnings
warnings.warn('this code is not up to date with the current data, see komanawa-slmacc-csra for the most recent version')

def plot_all_comps(correct=False):
    cor = ''
    if correct:
        cor = '_correct'
    outdir = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'norm', f'cumulative_hist_v_mod{cor}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for mode, site in default_mode_sites:
        sm = f'{site}-{mode}'
        for nyr in [1, 2, 3, 5, 10]:
            if nyr == 1:
                trend = pd.read_csv(
                    os.path.join(base_trend, f'{nyr}yr{cor}', f'{sm}_{nyr}yr_cumulative_exceed_prob.csv'))
                detrend = pd.read_csv(
                    os.path.join(base_detrend, f'{nyr}yr{cor}', f'{sm}_{nyr}yr_cumulative_exceed_prob.csv'))

            else:
                trend = pd.read_csv(os.path.join(base_trend, f'{nyr}yr-sequential{cor}',
                                                 f'{sm}_{nyr}yr_cumulative_exceed_prob.csv'))
                detrend = pd.read_csv(os.path.join(base_detrend, f'{nyr}yr-sequential{cor}',
                                                   f'{sm}_{nyr}yr_cumulative_exceed_prob.csv'))
                trend_non = pd.read_csv(os.path.join(base_trend, f'{nyr}yr-non-sequential{cor}',
                                                     f'{sm}_{nyr}yr_cumulative_exceed_prob.csv'))
                detrend_non = pd.read_csv(os.path.join(base_detrend, f'{nyr}yr-non-sequential{cor}',
                                                       f'{sm}_{nyr}yr_cumulative_exceed_prob.csv'))

            model = pd.read_csv(os.path.join(base_model, f'{nyr}yr{cor}', f'{sm}_{nyr}yr_cumulative_exceed_prob.csv'))

            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
            ax1.plot(trend.loc[:, 'pg'], 1 - trend.loc[:, 'prob'], c='r', label='historical (non-stationary) cdf')
            if nyr > 1:
                ax1.plot(trend_non.loc[:, 'pg'], 1 - trend_non.loc[:, 'prob'], c='m', label='historical (non-stationary) cdf - non-sequential')
            ax1.plot(detrend.loc[:, 'pg'], 1 - detrend.loc[:, 'prob'], c='b', label='detrended historical cdf')
            if nyr > 1:
                ax1.plot(detrend_non.loc[:, 'pg'], 1 - detrend_non.loc[:, 'prob'], c='c', label='detrended historical cdf - non-sequential')

            ax1.plot(model.loc[:, 'pg'], model.loc[:, 'prob'], c='k', label='modelled cdf')

            ax1.set_title('Exceedance probability')
            ax1.set_xlabel('Pasture growth tons DM/Ha/year')
            ax1.set_ylabel('Probability of an event with \nequal or greater Pasture growth')
            ax1.legend()

            ax2.plot(trend.loc[:, 'pg'], trend.loc[:, 'prob'], c='r', label='historical (non-stationary) cdf')
            if nyr > 1:
                ax2.plot(trend_non.loc[:, 'pg'], trend_non.loc[:, 'prob'], c='m', label='historical (non-stationary) cdf - non-sequential')
            ax2.plot(detrend.loc[:, 'pg'], detrend.loc[:, 'prob'], c='b', label='detrended historical cdf')
            if nyr > 1:
                ax2.plot(detrend_non.loc[:, 'pg'], detrend_non.loc[:, 'prob'], c='c', label='detrended historical cdf - non-sequential')

            ax2.plot(model.loc[:, 'pg'], 1 - model.loc[:, 'prob'], c='k', label='modelled cdf')

            ax2.set_title('Non-exceedance probability')
            ax2.set_xlabel('Pasture growth anomaly tons DM/Ha/year')
            ax2.set_ylabel('Probability of an event with \nequal or less Pasture growth')
            ax2.legend()

            nm = f'{site}-{mode}_exceed_prob'
            fig.suptitle(f'{site}-{mode} {nyr} year comparison'.capitalize())
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'{site}-{mode}_{nyr}yr_cum_comp.png'))


if __name__ == '__main__':
    plot_all_comps(correct=False)
    plot_all_comps(correct=True)
