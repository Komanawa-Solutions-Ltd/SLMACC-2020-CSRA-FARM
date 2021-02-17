"""
 Author: Matt Hanson
 Created: 17/02/2021 3:15 PM
 """
from BS_work.SWG.SWG_wrapper import _check_data_v1
import os
import pandas as pd

if __name__ == '__main__':  # todo start here with this problem
    swg_dirs = [
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base",
    ]
    storylines = [
        pd.read_csv(r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_csvs\test.csv'),
        pd.read_csv(r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_csvs\0-baseline.csv'),
    ]
    for swg_dir, story in zip(swg_dirs, storylines):
        print(swg_dir)
        num = []
        all_keys = []
        paths = pd.Series(os.listdir(swg_dir))
        paths = list(paths.loc[(~paths.str.contains('exsites')) & (paths.str.contains('.nc'))])
        for i, p in enumerate(paths):
            if i % 100 == 0:
                print('{} of {}'.format(i, len(paths)))
            nums, keys = _check_data_v1(os.path.join(swg_dir, p), story)
            num.append(nums)
            all_keys.extend(keys)
        out = pd.Series(num, index=paths)
        out.to_csv(os.path.join(swg_dir, 'number_diff.csv'))
        out2 = pd.Series(all_keys)
        out2.to_csv(os.path.join(swg_dir, 'dif_keys.csv'))
