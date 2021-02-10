"""
 Author: Matt Hanson
 Created: 10/02/2021 9:30 AM
 """
import os
import ksl_env
import numpy as np
import subprocess
import sys

oxford_lat, oxford_lon = -43.296, 172.192
swg = os.path.join(os.path.dirname(__file__), 'SWG_Final.py')
#todo check
def create_yaml(outpath_yml, out_data_base, nsims, storyline_path,
                base_dir=ksl_env.get_vscn_dir(), plat=172.333, plon=-43.372, xlat=None, xlon=None):
    SH_model_temp_file = os.path.join(os.path.dirname(__file__), 'SHTemps.dat')
    vcf = os.path.join(os.path.dirname(__file__), 'event_definition_data_fixed.csv')

    for d in [outpath_yml, out_data_base]:
        if not os.path.exists(os.path.dirname(d)):
            os.makedirs(d)

    exstat = False
    if xlat is not None or xlon is not None:
        assert xlat is not None
        assert xlon is not None
        exstat = True
        xlat = ''.join(['- {}\n'.format(e) for e in np.atleast1d(xlat)])
        xlon = ''.join(['- {}\n'.format(e) for e in np.atleast1d(xlon)])

    yml = ('# created with {create_file}'
           'base_directory: {bd}/ #Directory that contains the directory "SLMACC-Subset_vcsn", must have / at end'
           'SH_model_temp_file: {smtf} #Data file that includes the historic model run temperature data'
           'VCSN_Class_file: {vcf} #Data file that includes the historic classification of VCSN data'
           ''
           'lat_val: {plat} #latitudinal location of the primary site'
           'lon_val: {plon} #longitudinal location of the primary site'
           'month_scale_factor: 1 #A scale factor which increases month length, testing thing to make abitrarly long'
           'number_of_simulations: {n} #number of simulations requested'
           'story_line_filename: {slf} #Data file containing the storylines to be simulated'
           ''
           'simulation_savename: {savn} #base savename for the simulation output files'
           ''
           'netcdf_save_flag: True #Boolean where True requests that netcdf files are saved'
           ''
           'Extra_station_flag: {exstat} #Boolean where True requests that extra stations are simulated'
           'Extra_sim_savename: {exsvnm} #base savename for the simulation output files of any extra stations'
           'Extra_site_lat: #list of latitudinal locations of the extra sites'
           '{xlat}'
           'Extra_site_lon: #list of #longitudinal locations of the extra sites'
           '{xlon}'
           ).format(create_file=__file__,
                    bd=base_dir,
                    smtf=SH_model_temp_file,
                    vcf=vcf,
                    plat=plat,
                    plon=plon,
                    n=nsims,
                    slf=storyline_path,
                    savn=out_data_base,
                    exstat=exstat,
                    exsvnm=out_data_base + 'exsites',
                    xlat=xlat,
                    xlon=xlon,
                    )
    with open(outpath_yml,'w') as f:
        f.write(yml)


def run_SWG(yml_path): #todo check
    result = subprocess.run(sys.executable, swg, input=yml_path, stdout='PIPE',stderr='STDOUT')
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))
