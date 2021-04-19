"""
 Author: Matt Hanson
 Created: 20/04/2021 11:08 AM
 """

import ksl_env
import pandas as pd
import os
import glob
import shutil
from Storylines.storyline_building_support import default_mode_sites

if __name__ == '__main__':
    key = '2-yr1-Autumn Drough hot dry feb-March 50rest'
    output_dir = os.path.join(ksl_env.slmmac_dir, r"outputs_for_ws\Final_Storylines\Autumn_Drought")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_dir = os.path.join(ksl_env.slmmac_dir, r"outputs_for_ws\lauras_autum_drought_1yr")

    t = pd.read_csv(os.path.join(base_dir, 'IID_probs_pg.csv')).set_index('ID')
    mapper = {'prob': 'log10_prob_irrigated', 'pgr': 'oxford-irrigated_pg', 'pgra': 'oxford-irrigated_pgra'}
    t = t.rename(columns=mapper)
    t.loc[[key]].to_csv(os.path.join(output_dir, 'IID_probs_pg.csv'))

    for mode, site in default_mode_sites:
        od = os.path.join(output_dir, f'{site}-{mode}')
        pdir = os.path.join(od, 'plots')
        sdir = os.path.join(od, 'storylines')
        for d in [pdir, sdir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # copy plots
        for p in glob.glob(os.path.join(base_dir, f'{site}-{mode}', 'plots', f'*{key}*')):
            shutil.copyfile(p, os.path.join(pdir, os.path.basename(p)))

        # copy storylines
        for p in glob.glob(os.path.join(base_dir, f'{site}-{mode}', 'storylines', f'*{key}*')):
            shutil.copyfile(p, os.path.join(sdir, os.path.basename(p)))

        # copy data
        for p in glob.glob(os.path.join(base_dir, f'{site}-{mode}', '*.csv')):
            temp = pd.read_csv(p, index_col=0, header=[0, 1])
            temp.loc[[f'{key}-{site}-{mode}', f'0-baseline-{site}-{mode}']].to_csv(os.path.join(od, os.path.basename(p)))
