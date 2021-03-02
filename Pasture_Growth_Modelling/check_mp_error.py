"""
 Author: Matt Hanson
 Created: 1/03/2021 4:04 PM
 """
import os
import numpy as np

import sys

sys.path.append(['C:\\Users\\dumon\\python_projects\\SLMACC-2020-CSRA\\Pasture_Growth_Modelling',
                 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2020.1\\plugins\\python-ce\\helpers\\pydev',
                 'C:\\Users\\dumon\\python_projects\\SLMACC-2020-CSRA',
                 'C:\\Program Files\\DHI\\2017\\FEFLOW 7.1\\bin64',
                 'C:\\Users\\dumon\\python_projects\\SLMACC-2020-CSRA\\Pasture_Growth_Modelling',
                 'C:\\Users\\dumon\\python_projects\\SLMACC-2020-CSRA',
                 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2020.1\\plugins\\python-ce\\helpers\\third_party\\thriftpy',
                 'C:\\Program Files\\JetBrains\\PyCharm Community Edition 2020.1\\plugins\\python-ce\\helpers\\pydev',
                 'C:\\Users\\dumon\\AppData\\Local\\JetBrains\\PyCharmCE2020.1\\cythonExtensions',
                 'C:\\Users\\dumon\\python_projects\\SLMACC-2020-CSRA\\Pasture_Growth_Modelling',
                 'C:\\Users\\dumon\\.conda\\envs\\SLMMAC_20_CSRA3\\python37.zip',
                 'C:\\Users\\dumon\\.conda\\envs\\SLMMAC_20_CSRA3\\DLLs',
                 'C:\\Users\\dumon\\.conda\\envs\\SLMMAC_20_CSRA3\\lib',
                 'C:\\Users\\dumon\\.conda\\envs\\SLMMAC_20_CSRA3',
                 'C:\\Users\\dumon\\.conda\\envs\\SLMMAC_20_CSRA3\\lib\\site-packages'])
from Climate_Shocks.climate_shocks_env import storyline_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth, default_pasture_growth_dir
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp

if __name__ == '__main__':

    nsims = 10

    run_mp = True
    run_norm = False
    num = 20
    spaths = np.repeat([os.path.join(storyline_dir, '0-baseline.csv')], (num))
    odirs = [os.path.join(default_pasture_growth_dir, 'test_mp_f', f't{e}') for e in range(num)]

    if run_mp:
        run_full_model_mp(storyline_path_mult=spaths,
                          outdir_mult=odirs,
                          nsims_mult=num,
                          log_path=r"C:\Users\dumon\Downloads\test_log",
                          description_mult='just a test',
                          padock_rest_mult=True,
                          save_daily_mult=False,
                          verbose=False)
    if run_norm:
        for p, od in zip(spaths, odirs):
            print(p)
            run_pasture_growth(storyline_path=p, outdir=od, nsims=nsims, padock_rest=True,
                               save_daily=True, description='', verbose=True,
                               n_parallel=1)
