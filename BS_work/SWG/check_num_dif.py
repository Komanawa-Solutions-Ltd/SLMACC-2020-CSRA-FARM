"""
 Author: Matt Hanson
 Created: 17/02/2021 3:15 PM
 """
from BS_work.SWG.SWG_wrapper import check_data_v1
import os
import pandas as pd


def statsof_data(base_dir, ex=''):
    test = pd.read_csv(os.path.join(base_dir, "{}_dif_keys.csv".format(ex)))
    t = test.groupby('0').count()
    t.to_csv(os.path.join(base_dir, "{}_dif_keys_count.csv".format(ex)))
    test = test.loc[:, '0']
    t = pd.Series(e.split(':')[1] for e in test)
    t.groupby(t).count().to_csv(os.path.join(base_dir, "{}_dif_keys_change.csv".format(ex)))
    t = pd.Series(e.split(':')[0] for e in test)
    t.groupby(t).count().to_csv(os.path.join(base_dir, "{}_dif_months.csv".format(ex)))

def classify_bad_data(storyline_path, swg_dir, check_fun=check_data_v1, ex=''):
    """

    :param storyline_path:
    :param swg_dir:
    :param check_fun: differnt functions
    :param ex: string to append to file names.
    :return:
    """
    story = pd.read_csv(storyline_path)
    print(swg_dir)
    num = []
    all_keys = []
    paths = pd.Series(os.listdir(swg_dir))
    paths = list(paths.loc[(~paths.str.contains('exsites')) & (paths.str.contains('.nc'))])
    for i, p in enumerate(paths):
        if i % 100 == 0:
            print('{} of {}'.format(i, len(paths)))
        nums, keys = check_fun(os.path.join(swg_dir, p), story)
        num.append(nums)
        all_keys.extend(keys)
    out = pd.Series(num, index=paths)
    out.to_csv(os.path.join(swg_dir, '{}_number_diff.csv'.format(ex)))
    out2 = pd.Series(all_keys)
    out2.to_csv(os.path.join(swg_dir, '{}_dif_keys.csv'.format(ex)))
    statsof_data(swg_dir, ex)


if __name__ == '__main__':
    # problem should be solved
    run_checks =True

    if run_checks:
        swg_dirs = [
            r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base",
            r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test",
        ]
        storylines = [
            pd.read_csv(r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_csvs\0-baseline.csv'),
            pd.read_csv(r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_csvs\test.csv'),
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
                nums, keys = check_data_v1(os.path.join(swg_dir, p), story)
                num.append(nums)
                all_keys.extend(keys)
            out = pd.Series(num, index=paths)
            out.to_csv(os.path.join(swg_dir, 'number_diff.csv'))
            out2 = pd.Series(all_keys)
            out2.to_csv(os.path.join(swg_dir, 'dif_keys.csv'))
            statsof_data(swg_dir)
