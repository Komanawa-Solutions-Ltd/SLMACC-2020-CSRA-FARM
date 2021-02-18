"""
 Author: Matt Hanson
 Created: 18/02/2021 10:03 AM
 """
from Climate_Shocks.climate_shocks_env import storyline_dir
from BS_work.SWG.SWG_wrapper import *
from BS_work.SWG.SWG_wrapper import _check_data_v1, _check_data_v2
from BS_work.SWG.check_num_dif import classify_bad_data

if __name__ == '__main__':
    # todo set up the check and deletion of any data that does not fit limits
    # todo run on dickie
    run_swg=False
    outdirs = [
        os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', 'test_detrend', 'test'),
        os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', 'test_detrend', 'base')
    ]

    storyline_nms = [
        'test.csv',
        '0-baseline.csv'
    ]

    detrended_vcf = r"D:\SLMMAC_SWG_test_detrend\detrend_event_data_fixed.csv"
    # run swg
    for outdir, sfilname in zip(outdirs, storyline_nms):
        if run_swg:
            print('running SWG')
            yml = os.path.join(outdir, 'test.yml')
            create_yaml(outpath_yml=yml, outsim_dir=outdir,
                        nsims=100,
                        storyline_path=os.path.join(storyline_dir, sfilname),
                        base_dir=r"D:\SLMMAC_SWG_test_detrend",
                        sim_name=None,
                        xlat=oxford_lat, xlon=oxford_lon,
                        vcf=detrended_vcf)
            temp = run_SWG(yml, outdir, rm_npz=True, clean=False)
            print(temp)
        classify_bad_data(os.path.join(storyline_dir, sfilname), outdir, check_fun=_check_data_v2,ex='w2tol')
