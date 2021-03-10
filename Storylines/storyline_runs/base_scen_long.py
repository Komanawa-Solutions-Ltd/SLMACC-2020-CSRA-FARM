"""
 Author: Matt Hanson
 Created: 16/02/2021 12:55 PM
 """
import netCDF4 as nc
from Climate_Shocks.climate_shocks_env import storyline_dir
from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth, default_pasture_growth_dir
from Pasture_Growth_Modelling.plot_full_model import plot_sims
from BS_work.SWG.SWG_wrapper import *

if __name__ == '__main__':
    run_basgra = False  # to stop accidental re-run
    plot_results = True

    if run_basgra:
        # run basgra
        print('running BASGRA')
        run_pasture_growth(storyline_path=os.path.join(storyline_dir, '0-long-baseline.csv'),
                           outdir=os.path.join(default_pasture_growth_dir, 'long_baseline'),
                           nsims=1000, padock_rest=False,
                           save_daily=False, description='initial long baseline run to help set ibasal',
                           verbose=True, fix_leap=True)

    path_list = [
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\long_baseline\0-long-baseline-eyrewell-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\long_baseline\0-long-baseline-oxford-irrigated.nc",
            r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\long_baseline\0-long-baseline-oxford-dryland.nc"
        ]
    out = {}
    for p in path_list:
        test = nc.Dataset(p)
        idx = (np.array(test.variables['m_year']) >2040) & (np.array(test.variables['m_month']) ==7)
        t =np.array(test.variables['m_BASAL'][idx]).mean()
        out[os.path.basename(p)] = t
        print(os.path.basename(p),':',t)

    if plot_results:
        plot_sims(data_paths=path_list[0:2], plot_ind=False, nindv=50, save_dir=None, show=False, figsize=(11, 8),
                  daily=False
                  )
        plot_sims(data_paths=path_list[-1:], plot_ind=False, nindv=50, save_dir=None, show=True, figsize=(11, 8),
                  daily=False
                  )

        # what to do about dryland IBASAL, leave it as is


