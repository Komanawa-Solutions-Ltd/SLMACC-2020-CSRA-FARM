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


# todo check
def create_yaml(outpath_yml, outsim_dir, nsims, storyline_path, sim_name=None,
                base_dir=os.path.dirname(ksl_env.get_vscn_dir()),
                plat=-43.372, plon=172.333, xlat=None, xlon=None):
    """
    create the yml file to run the stochastic weather generator: data will be created in outsim_dir/simname{} files
    :param outpath_yml: outpath to the YML file
    :param outsim_dir: outdir for the simulations
    :param nsims: number of simulations to create (note file per sim
    :param storyline_path: path to the storyline csv
    :param sim_name: if None, then the csv name, else other name
    :param base_dir: Directory that contains the directory "SLMACC-Subset_vcsn", default should be fine
    :param plat: primary lat (eyrewell)
    :param plon: primary lon (eyrewell)
    :param xlat: extra lats float or list like
    :param xlon: extra lons float or list like
    :return:
    """
    if sim_name is None:
        sim_name = os.path.splitext(os.path.basename(storyline_path))[0]
    SH_model_temp_file = os.path.join(os.path.dirname(__file__), 'SHTemps.dat')
    vcf = os.path.join(os.path.dirname(__file__), 'event_definition_data_fixed.csv')

    for d in [os.path.dirname(outpath_yml), outsim_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    exstat = False
    if xlat is not None or xlon is not None:
        assert xlat is not None
        assert xlon is not None
        exstat = True
        xlat = ''.join(['- {}\n'.format(e) for e in np.atleast_1d(xlat)])
        xlon = ''.join(['- {}\n'.format(e) for e in np.atleast_1d(xlon)])

    yml = ('# created with {create_file}\n'
           'base_directory: {bd}/ #Directory that contains the directory "SLMACC-Subset_vcsn", must have / at end\n'
           'SH_model_temp_file: {smtf} #Data file that includes the historic model run temperature data\n'
           'VCSN_Class_file: {vcf} #Data file that includes the historic classification of VCSN data\n'
           '\n'
           'lat_val: {plat} #latitudinal location of the primary site\n'
           'lon_val: {plon} #longitudinal location of the primary site\n'
           'month_scale_factor: 1 #A scale factor which increases month length, testing thing to make abitrarly long\n'
           'number_of_simulations: {n} #number of simulations requested\n'
           'story_line_filename: {slf} #Data file containing the storylines to be simulated\n'
           '\n'
           'simulation_savename: {savn} #base savename for the simulation output files\n'
           '\n'
           'netcdf_save_flag: True #Boolean where True requests that netcdf files are saved\n'
           '\n'
           'Extra_station_flag: {exstat} #Boolean where True requests that extra stations are simulated\n'
           'Extra_sim_savename: {exsvnm} #base savename for the simulation output files of any extra stations\n'
           'Extra_site_lat: #list of latitudinal locations of the extra sites\n'
           '{xlat}\n'
           'Extra_site_lon: #list of #longitudinal locations of the extra sites\n'
           '{xlon}\n'
           ).format(create_file=__file__,
                    bd=base_dir,
                    smtf=SH_model_temp_file,
                    vcf=vcf,
                    plat=plat,
                    plon=plon,
                    n=nsims,
                    slf=storyline_path,
                    savn=os.path.join(outsim_dir, sim_name),
                    exstat=exstat,
                    exsvnm=os.path.join(outsim_dir, sim_name + 'exsites'),
                    xlat=xlat,
                    xlon=xlon,
                    )
    with open(outpath_yml, 'w') as f:
        f.write(yml)


def run_SWG(yml_path):  # todo check
    result = subprocess.run([sys.executable, swg, yml_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))
    return '{}, {}'.format(result.stdout, result.stderr)


if __name__ == '__main__':
    outdir = r"C:\Users\Matt Hanson\Downloads\test_SWG"
    yml = os.path.join(outdir, 'test.yml')
    create_yaml(outpath_yml=yml, outsim_dir=outdir,
                nsims=10, storyline_path=os.path.join(os.path.dirname(__file__), 'v7.csv'),
                sim_name=None,
                xlat=oxford_lat, xlon=oxford_lon)
    temp = run_SWG(yml)
    print(temp)
    # todo check that these worked... should have
