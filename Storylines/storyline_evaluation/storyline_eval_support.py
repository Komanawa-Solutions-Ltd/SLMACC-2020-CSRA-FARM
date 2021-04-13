"""
 Author: Matt Hanson
 Created: 13/04/2021 9:42 AM
 """
import os
import pandas as pd
import numpy as np
from Storylines.storyline_building_support import base_events, climate_shocks_env, month_len
from BS_work.IID.IID import run_IID


def get_pgr_prob_baseline_stiched(nyears, site, mode,irr_prop_from_zero, recalc=False):
    """
    this uses the addition method
    :param nyears:
    :return:
    """
    if irr_prop_from_zero:
        t = 'irr_from_0'
    else:
        t = 'irr_from_50'
    save_path = os.path.join(climate_shocks_env.supporting_data_dir, 'baseline_data',
                             f'base_stiched{site}_{mode}_{t}.npy')
    # todo does the baseline prob need to be handled for non irrigation months

    # make storyline
    if not recalc and os.path.exists(save_path):
        prob, pgr = np.load(save_path)

    else:
        months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        storyline = pd.DataFrame(columns=['precip_class', 'temp_class', 'rest'])
        storyline.loc[:, 'month'] = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        storyline.loc[:, 'year'] = [2024, 2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025, 2025, 2025, 2025,]
        for i, m in storyline.loc[:, ['month']].itertuples(True, None):
            temp, precip, rest = base_events[m]
            storyline.loc[i, 'precip_class'] = precip
            storyline.loc[i, 'temp_class'] = temp
            storyline.loc[i, 'rest'] = rest
        prob = run_IID({'base': storyline},
                       irr_prob_from_zero=irr_prop_from_zero).set_index('ID').loc['base', 'log10_prob']
        temp = pd.read_csv(os.path.join(climate_shocks_env.supporting_data_dir,
                                        f'baseline_data/{site}-{mode}-monthly.csv'),
                           skiprows=1
                           ).set_index(['year', 'sim_mon_day'])
        pgr = np.sum([temp.loc[(y, m), 'PGR'] * month_len[m] for y,m in storyline.loc[:,['year',
                                                                                         'month']].itertuples(False,
                                                                                                              None)])
        np.save(save_path, [prob, pgr], False)

    outprob, outpgr = 0, 0
    for i in range(nyears):
        outprob += prob
        outpgr += pgr
    return outpgr, outprob


if __name__ == '__main__':
    print(get_pgr_prob_baseline_stiched(1,'eyrewell','irrigated', True, True))
    print(get_pgr_prob_baseline_stiched(1,'oxford','irrigated', True, True))
    print(get_pgr_prob_baseline_stiched(1,'oxford','dryland', True, True))
    print(get_pgr_prob_baseline_stiched(1,'eyrewell','irrigated', False, True))
    print(get_pgr_prob_baseline_stiched(1,'oxford','irrigated', False, True))
    print(get_pgr_prob_baseline_stiched(1,'oxford','dryland', False, True))
