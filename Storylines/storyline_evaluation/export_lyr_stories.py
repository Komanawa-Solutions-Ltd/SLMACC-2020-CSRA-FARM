"""
 Author: Matt Hanson
 Created: 19/04/2021 9:00 AM
 """
import shutil
import project_base
import os
import pandas as pd
import glob
import numpy as np
from Storylines.storyline_runs.run_random_suite import get_1yr_data, default_mode_sites, random_sl_dir


def export_1yr_stories(output_dir, n, anamoly, site, mode, tolerance):
    """
    pull the top n probable sites
    :param n:
    :param anamoly: - for deficit + for surplus in kg DM/ha/year
    :param site:
    :param mode:
    :param tolerance: absolute tolerance to use with np.isclose
    :return:
    """
    d = os.path.join(output_dir, f'{site}-{mode}')
    if not os.path.exists(d):
        os.makedirs(d)
    else:
        shutil.rmtree(d)
        while os.path.exists(d):  # check if it exists
            pass
        os.makedirs(d)

    with open(os.path.join(output_dir, f'{site}-{mode}', 'readme.txt'), 'w') as f:
        f.write(f'data pulled for anamoly of {anamoly} kgDM/ha/yr with tolerance of +- {tolerance}\n')
        f.write(f'top_dataset contains all of the PG and probability data for the copied storylines\n')
        f.write(f'full_dataset contains all of the PG and probability data for all 1yr storylines\n')
        f.write('storylines file names are in the following format "00000-rsl-054011_good-irr"\n'
                '    where "00000" = rank for probability 0= most probable,\n'
                '    "rsl-054011" = random storyline identifier,\n'
                '    good-irr = restriction options, \n'
                '               good = 5-50 percentile restrictions and\n'
                '               bad = 50-95 percentile restrictions.')

    data = get_1yr_data()
    data.to_csv(os.path.join(output_dir, f'{site}-{mode}', 'full_dataset.csv'))
    use_data = data.loc[np.isclose(data.loc[:, f'{site}-{mode}_pgra_yr1'], anamoly, atol=tolerance)]
    use_data = use_data.loc[use_data.irr_type != 'baseline']
    use_data = use_data.sort_values(f'log10_prob_{mode}', ascending=False).iloc[:n]
    use_data.to_csv(os.path.join(output_dir, f'{site}-{mode}', 'top_dataset.csv'))

    for i, (idv, irr_type) in enumerate(use_data[['ID', 'irr_type']].itertuples(False, None)):
        shutil.copyfile(src=os.path.join(f'{random_sl_dir}_{irr_type}_irr', f'{idv}.csv'),
                        dst=os.path.join(output_dir, f'{site}-{mode}', f'{i:05d}-{idv}_{irr_type}-irr.csv'))

    # sumerize
    paths = glob.glob(os.path.join(output_dir, f'{site}-{mode}', f'*-irr.csv'))
    counts_with_irr = {e: {} for e in range(1, 13)}
    counts_wo_irr = {e: {} for e in range(1, 13)}
    for p in paths:
        data = pd.read_csv(p).set_index('month')
        for m in range(1, 13):
            t = counts_with_irr[m]
            k = f'{data.loc[m, "precip_class"]}P-{data.loc[m, "temp_class"]}T-{data.loc[m, "rest_per"]}R'
            if k in t.keys():
                t[k] += 1
            else:
                t[k] = 1
            t = counts_wo_irr[m]
            k = f'{data.loc[m, "precip_class"]}P-{data.loc[m, "temp_class"]}T'
            if k in t.keys():
                t[k] += 1
            else:
                t[k] = 1
    counts_with_irr = pd.DataFrame(counts_with_irr)
    counts_wo_irr = pd.DataFrame(counts_wo_irr)
    counts_with_irr.to_csv(os.path.join(d, 'counts_with_irrigation.csv'))
    counts_wo_irr.to_csv(os.path.join(d, 'counts_without_irrigation.csv'))
    # sl.to_csv(os.path.join(f'{random_sl_dir}{tnm}', f'{name}.csv')) directory...


if __name__ == '__main__':
    output_dir = os.path.join(project_base.slmmac_dir, 'outputs_for_ws', 'top_prob_1yr_sl')
    for mode, site in default_mode_sites:
        export_1yr_stories(output_dir, 100, -2500, site, mode, 500)
