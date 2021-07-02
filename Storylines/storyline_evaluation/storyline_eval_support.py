"""
 Author: Matt Hanson
 Created: 13/04/2021 9:42 AM
 """
import os
import pandas as pd
import netCDF4 as nc
import glob
import numpy as np
from Storylines.storyline_building_support import base_events, climate_shocks_env, month_len
from Pasture_Growth_Modelling.full_model_implementation import default_mode_sites
from BS_work.IID.IID import run_IID
from Pasture_Growth_Modelling.historical_average_baseline import get_historical_average_baseline


def get_pgr_prob_baseline_stiched_old(nyears, site, mode, irr_prop_from_zero, recalc=False):
    """
    this uses the addition method
    :param nyears:
    :return:
    """
    raise ValueError('old version, depreciated')
    add_irr_prob = True
    if mode == 'dryland':
        add_irr_prob = False
    if irr_prop_from_zero:
        t = 'irr_from_0'
    else:
        t = 'irr_from_50'
    save_path = os.path.join(climate_shocks_env.supporting_data_dir, 'baseline_data',
                             f'base_stiched{site}_{mode}_{t}.npy')

    # make storyline
    if not recalc and os.path.exists(save_path):
        prob, pgr = np.load(save_path)

    else:
        months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        storyline = pd.DataFrame(columns=['precip_class', 'temp_class', 'rest'])
        storyline.loc[:, 'month'] = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        storyline.loc[:, 'year'] = [2024, 2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025, 2025, 2025, 2025, ]
        for i, m in storyline.loc[:, ['month']].itertuples(True, None):
            temp, precip, rest, rest_per = base_events[m]
            storyline.loc[i, 'precip_class'] = precip
            storyline.loc[i, 'temp_class'] = temp
            storyline.loc[i, 'rest'] = rest
            storyline.loc[i, 'rest_per'] = rest_per
        prob = run_IID({'base': storyline},
                       irr_prob_from_zero=irr_prop_from_zero,
                       add_irr_prob=add_irr_prob).set_index('ID').loc['base', 'log10_prob']
        temp = pd.read_csv(os.path.join(climate_shocks_env.supporting_data_dir,
                                        f'baseline_data/{site}-{mode}-monthly.csv'),
                           skiprows=1
                           ).set_index(['year', 'sim_mon_day'])
        pgr = np.sum([temp.loc[(y, m), 'PGR'] * month_len[m] for y, m in storyline.loc[:, ['year',
                                                                                           'month']].itertuples(False,
                                                                                                                None)])
    outprob, outpgr = 0, 0
    for i in range(nyears):
        outprob += prob
        outpgr += pgr
    return outpgr, outprob


def get_pgr_prob_baseline_stiched(nyears, site, mode):  # using historical baseline so no prob values.
    """
    this uses the addition method
    :param nyears:
    :return:
    """

    pgr, date_run = get_historical_average_baseline(site, mode, [2024])  # just one year of data
    pgr = pgr.PGR.sum()

    outpgr = 0
    for i in range(nyears):
        outpgr += pgr
    return outpgr


def calc_impact_prob(pgr, prob, stepsize=0.1, normalize=True):
    """
    chunk and sum probability based on the pgr, in steps
    :param pgr:
    :param prob:
    :param stepsize:
    :param normalize: bool if True then make sum of out_prob = 1
    :return:
    """
    pgr = np.atleast_1d(pgr)
    prob = np.atleast_1d(prob)

    step_decimals = len(str(stepsize).split('.')[-1])
    min_val = np.round(np.min(pgr) - stepsize, step_decimals)
    max_val = np.round(np.max(pgr) + stepsize, step_decimals)
    steps = np.arange(min_val, max_val + stepsize / 2, stepsize / 2)
    out_prob = np.zeros(len(steps) - 1)
    out_pgr = np.zeros(len(steps) - 1)
    mx_prob = np.ceil(prob).max()
    for i, (l, u) in enumerate(zip(steps[0:-1], steps[1:])):
        out_pgr[i] = np.nanmean([l, u])
        idx = (pgr >= l) & (pgr < u)
        if normalize:
            out_prob[i] = np.nansum(10 ** (prob[idx]-mx_prob))
        else:
            out_prob[i] = np.nansum(10 ** (prob[idx]))
    if normalize:
        out_prob *= 1 / out_prob.sum()
    return out_pgr, out_prob


def calc_cumulative_impact_prob(pgr, prob, stepsize=0.1, more_production_than=True):
    """
    calc the exceedence probaility
    :param pgr:
    :param prob:
    :param stepsize:
    :param more_production_than: bool if True the probability that an event with pgr as equal to or higher than the value
                                 else, the probability that an event with pgr equal to or lower than the value
    :return:
    """
    im_pgr, im_prob = calc_impact_prob(pgr, prob, stepsize, True)

    out_prob = np.zeros(im_pgr.shape)
    if more_production_than:
        for i, v in enumerate(im_pgr):
            out_prob[i] = im_prob[im_pgr >= v].sum()
    else:
        for i, v in enumerate(im_pgr):
            out_prob[i] = im_prob[im_pgr <= v].sum()
    return im_pgr, out_prob


def extract_additional_sims(story_dir, sim_dir, nyr):
    storylines = glob.glob(os.path.join(story_dir, '*.csv'))
    # run IID
    data = run_IID(story_dict={os.path.basename(v).replace('.csv', ''): pd.read_csv(v) for v in storylines},
                   verbose=False,
                   irr_prob_from_zero=False, add_irr_prob=True)
    data.set_index('ID', inplace=True)
    data.rename(columns={'log10_prob': 'log10_prob_irrigated'}, inplace=True)

    temp = run_IID(story_dict={os.path.basename(v).replace('.csv', ''): pd.read_csv(v) for v in storylines},
                   verbose=False,
                   irr_prob_from_zero=False, add_irr_prob=False).set_index('ID')
    data.loc[:, 'log10_prob_dryland'] = temp.loc[:, 'log10_prob']
    data.reset_index(inplace=True)

    for mode, site in default_mode_sites:
        key = f'{site}-{mode}'
        data.loc[:, f'{key}_pg'] = np.nan
        data.loc[:, f'{key}_pgra'] = np.nan
        for i, idv in data.loc[:, ['ID']].itertuples(True, None):
            if i % 10 == 0:
                print(f'starting to read sim {i} for site: {site} and mode: {mode}')
            p = os.path.join(sim_dir, f'{idv}-{key}.nc')

            try:
                nc_data = nc.Dataset(p)
                data.loc[i, f'{key}_pgra'] = np.array(nc_data.variables['m_PGRA_cum'][-1, :]).mean()
                temp = np.array(nc_data.variables['m_PGR'])
                temp *= np.repeat([31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30], nyr)[:, np.newaxis]
                temp = temp.sum(axis=0).mean()
                data.loc[i, f'{key}_pg'] = temp
                nc_data.close()
            except FileNotFoundError:
                pass
    data.to_csv(os.path.join(sim_dir, f'IID_probs_pg.csv'))
    return data


if __name__ == '__main__':
    print(get_pgr_prob_baseline_stiched(1, 'eyrewell', 'irrigated'))
    print(get_pgr_prob_baseline_stiched(1, 'oxford', 'irrigated'))
    print(get_pgr_prob_baseline_stiched(1, 'oxford', 'dryland'))
