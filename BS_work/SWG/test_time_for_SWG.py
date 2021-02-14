"""
 Author: Matt Hanson
 Created: 12/02/2021 10:33 AM
 """
from BS_work.SWG.SWG_wrapper import *
import itertools
import pandas as pd
import time

if __name__ == '__main__':
    numbers = [1, 10, 100, 1000, 10000]
    numbers = [1]
    base_dir = r"C:\Users\Matt Hanson\Downloads"
    outdata = pd.DataFrame(columns=['remove_npz',
                                    'number',
                                    'time_per_sim',
                                    'total_time'])

    for i, (n, rm) in enumerate(itertools.product(numbers, [True])):
        t = '_rm_npz'
        if not rm:
            t = ''
        name = '{}{}'.format(n, t)
        print(name)
        t = time.time()
        outdir = os.path.join(base_dir, name)
        yml = os.path.join(outdir, 'test.yml')
        create_yaml(outpath_yml=yml, outsim_dir=outdir,
                    nsims=n,
                    storyline_path=r'C:\Users\Matt Hanson\python_projects\SLMACC-2020-CSRA\BS_work\SWG\v7.csv',
                    sim_name=None,
                    xlat=oxford_lat, xlon=oxford_lon)
        temp = run_SWG(yml, outdir, rm_npz=rm)
        outdata.loc[i, 'remove_npz'] = rm
        outdata.loc[i, 'number'] = n
        outdata.loc[i, 'time_per_sim'] = (time.time() - t) / n
        outdata.loc[i, 'total_time'] = (time.time() - t)

    temp = run_SWG(yml, outdir, rm_npz=rm)
    outdata.to_csv(os.path.join(r"M:\Shared drives\Z2003_SLMACC", 'time_test_SWG.csv')) # todo run on dickie
