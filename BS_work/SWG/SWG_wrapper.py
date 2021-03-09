"""
 Author: Matt Hanson
 Created: 10/02/2021 9:30 AM
 """
import os
import ksl_env
import numpy as np
import datetime
import subprocess
import sys
import netCDF4 as nc
import glob
import yaml
import pandas as pd
import shutil
from Climate_Shocks.Stochastic_Weather_Generator.read_swg_data import read_swg_data
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import detrended_start_month, calc_smd_monthly
from Climate_Shocks.get_past_record import get_vcsn_record
from Climate_Shocks import climate_shocks_env
from Storylines.check_storyline import get_acceptable_events

oxford_lat, oxford_lon = -43.296, 172.192
swg = os.path.join(os.path.dirname(__file__), 'SWG_Final.py')
default_vcf = os.path.join(climate_shocks_env.supporting_data_dir, 'event_definition_data_fixed.csv')
default_base_dir = os.path.dirname(ksl_env.get_vscn_dir())


# note that each sim takes c. 0.12 mb of storage space.

def create_yaml(outpath_yml, outsim_dir, nsims, storyline_path, sim_name=None,
                base_dir=default_base_dir,
                plat=-43.372, plon=172.333, xlat=None, xlon=None, vcf=default_vcf):
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
    :param vcf: Data file that includes the historic classification of VCSN data
    :return:
    """
    assert '_' not in os.path.basename(storyline_path), '_ in file basename mucks things up'
    if sim_name is None:
        sim_name = os.path.splitext(os.path.basename(storyline_path))[0]
    SH_model_temp_file = os.path.join(os.path.dirname(__file__), 'SHTemps.dat')

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
    else:
        xlat = '- \n'
        xlon = '- \n'

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


def run_SWG(yml_path, outdir, rm_npz=True):
    """
    run the SWG
    :param yml_path: path to the yml file
    :param outdir: output directory (only needed if deleting .npz files
    :param rm_npz: boolean if true, remove all the .npz files in the folder to save space
    :return:
    """
    result = subprocess.run([sys.executable, swg, yml_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    if rm_npz:
        temp = glob.glob(os.path.join(outdir, '*.npz'))
        for f in temp:
            os.remove(f)
    return '{}, {}'.format(result.stdout, result.stderr)


def clean_swg(swg_dir, yml_path, duplicate=True, merge=True, nc_outpath=None):
    """
    remove swg files which do not match the data. Note that definitions are hard coded into this process
    via _check_data_v1.
    :param swg_dir: directory for the swg
    :param yml_path: path to the yml file
    :param exsites: int, number of external sites
    :param duplicate: boolean if True duplicate before removing bad files.
    :param merge: boolean if True merge everything into one large nc file
    :param nc_outpath: the path to save the merged path if merged is True
    :return:
    """
    if merge:
        assert nc_outpath is not None, ' if merging must define nc outpath'

    if duplicate:
        base_dup_dir = os.path.join(os.path.dirname(os.path.dirname(swg_dir)),
                                    '{}_duplicated'.format(os.path.basename(os.path.dirname(swg_dir))))
        if not os.path.exists(base_dup_dir):
            os.makedirs(base_dup_dir)
        shutil.copytree(swg_dir, os.path.join(base_dup_dir, os.path.basename(swg_dir)))

    with open(yml_path, 'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_dict = yaml.load(file, Loader=yaml.FullLoader)
    storyline_path = param_dict['story_line_filename']
    storyline = pd.read_csv(storyline_path)
    paths = pd.Series(sorted(os.listdir(swg_dir)))
    paths = paths.loc[paths.str.contains('.nc')]

    sites = (['exsites_{}'.format(e) for e in
              paths.loc[paths.str.contains('exsites')].str.split('_').str[1].unique()])

    paths = paths.loc[~paths.str.contains('exsites')]
    removed = []
    cold_months, wet_months, hot_months, dry_months = _get_possible_months()
    for p in paths:
        m = int(p.split('-')[0].replace('m', ''))
        fp = os.path.join(swg_dir, p)
        bad = _check_data_v1(fp, storyline, m, cold_months, wet_months, hot_months, dry_months)
        if bad:
            removed.append(p)
            for i in range(len(sites)):
                vals = p.split('_')
                if len(vals) > 2:
                    raise ValueError('names preclude removal of exsites')
                os.remove(os.path.join(swg_dir, '{}exsites_P{}_{}'.format(vals[0], i, vals[1])))
            os.remove(fp)
    with open(yml_path, 'r') as file:
        yml_text = file.readlines()
    _merge_ncs(swg_dir, nc_outpath, storyline, yml_txt=yml_text)
    return removed


def _get_possible_months():
    cold_months, wet_months, hot_months, dry_months = [], [], [], []
    acceptable_events = get_acceptable_events()
    for k, v in acceptable_events.items():
        if 'C' in k:
            cold_months.extend(v)
        if 'H' in k:
            hot_months.extend(v)
        if 'W' in k:
            wet_months.extend(v)
        if 'D' in k:
            dry_months.extend(v)

        cold_months = list(set(cold_months))
        wet_months = list(set(wet_months))
        hot_months = list(set(hot_months))
        dry_months = list(set(dry_months))
    return cold_months, wet_months, hot_months, dry_months


def _check_data_v1(swg_path, storyline, m, cold_months, wet_months, hot_months, dry_months,
                   return_full_results=False):
    """
    check that a single realisation is correct
    :param swg_path: path to the SWG
    :param yml_path: path to the YML
    :param m: None or int or list of int, months to check
    :return: True
    """

    storyline = storyline.loc[np.in1d(storyline.month, m)]
    storyline = storyline.set_index(['year', 'month'])

    data = read_swg_data(swg_path)[0]
    data = data.loc[np.in1d(data.month, m)]

    # calc SMA
    data.loc[:, 'sma'] = calc_smd_monthly(data.rain, data.pet, data.index) - data.loc[:, 'doy'].replace(
        get_monthly_smd_mean_detrended())

    data.loc[:, 'wet'] = data.loc[:, 'rain'] >= 0.1
    data.loc[:, 'dry'] = data.loc[:, 'sma'] <= -15
    data.loc[:, 'hot'] = data.loc[:, 'tmax'] >= 25
    data.loc[:, 'cold'] = ((data.loc[:, 'tmin'] +
                            data.loc[:, 'tmax']) / 2).rolling(3).mean().fillna(method='bfill') <= 7

    temp = data.loc[:, ['year', 'month', 'wet',
                        'dry', 'hot', 'cold']].groupby(['year', 'month']).sum()
    storyline.loc[temp.index, ['wet', 'dry', 'hot', 'cold']] = temp.loc[:, ['wet', 'dry', 'hot', 'cold']]
    storyline.reset_index(inplace=True)

    storyline.loc[:, 'swg_precip_class'] = 'A'
    storyline.loc[((storyline.wet >= storyline.month.replace(rain_limits_wet)) &
                   np.in1d(storyline.month, wet_months)), 'swg_precip_class'] = 'W'
    # dry out weighs wet if both happen
    storyline.loc[((storyline.dry >= 10) &
                   np.in1d(storyline.month, dry_months)), 'swg_precip_class'] = 'D'

    storyline.loc[:, 'swg_temp_class'] = 'A'
    storyline.loc[(storyline.hot >= 7) & np.in1d(storyline.month, hot_months), 'swg_temp_class'] = 'H'
    storyline.loc[(storyline.cold >= 10) & np.in1d(storyline.month, cold_months), 'swg_temp_class'] = 'C'

    num_dif = (~((storyline.temp_class == storyline.swg_temp_class) & (
            storyline.precip_class == storyline.swg_precip_class))).sum()
    if not return_full_results:
        return num_dif > 0

    hot = storyline.hot.max()
    cold = storyline.cold.max()
    wet = storyline.wet.max()
    dry = storyline.dry.max()

    where_same = ((storyline.temp_class == storyline.swg_temp_class) & (
            storyline.precip_class == storyline.swg_precip_class))

    out_keys = ['{}:{}-{}_{}-{}'.format(m, p, swgp, t, swgt) for m, p, swgp, t, swgt in
                storyline.loc[~where_same, ['month',
                                            'precip_class',
                                            'swg_precip_class',
                                            'temp_class',
                                            'swg_temp_class'
                                            ]].itertuples(False,
                                                          None)]
    return num_dif, out_keys, hot, cold, wet, dry


def _merge_ncs(swg_dir, outpath, storyline, yml_txt):
    """
    merge all of the ncs into a single netcdf file no cleaning takes place
    :param swg_dir: dir to the individual swg
    :param outpath: path
    :return:
    """
    out = nc.Dataset(outpath, 'w')

    paths = pd.Series(sorted(os.listdir(swg_dir)))
    paths = paths.loc[(paths.str.contains('.nc'))]
    nreal = (~paths.str.contains('exsites')).sum()
    reals = [int(e) for e in
             paths.loc[(~paths.str.contains('exsites'))].str.split('_').str[-1].str.replace('S', '').str.replace('.nc',
                                                                                                                 '')]
    if nreal == 0:
        print('no realisations for {}'.format(swg_dir))
        return None
    temp = nc.Dataset(os.path.join(swg_dir, paths.iloc[0]))

    # create dimensions and dimension variables
    days = temp.dimensions['day'].size
    out.createDimension('day', days)
    day = out.createVariable('day', int, ('day',), fill_value=-1)
    day[:] = range(days)
    day.setncatts({'long_name': 'day of simulation'})

    month = out.createVariable('month', int, ('day',), fill_value=-1)
    month[:] = np.array(temp.variables['Month'])
    month.setncatts({'long_name': 'month of simulation'})

    out.createDimension('real', nreal)
    real = out.createVariable('real', int, ('real',), fill_value=-1)
    real[:] = range(nreal)
    real.setncatts({'long_name': 'the realisation number of the simulation'})

    real = out.createVariable('org_real', int, ('real',), fill_value=-1)
    real[:] = reals
    real.setncatts({'long_name': 'the original realisation number of the simulation from when the SWG '
                                 'was run and before any filtering'})

    measures = ['PR_A', 'Tmax', 'Tmin', 'RSDS', 'PEV']
    out.createDimension('w_var', len(measures))
    ms = out.createVariable('w_var', str, ('w_var',))
    for i, m in enumerate(measures):
        ms[i] = m
    ms.setncatts({e: [temp.variables[m].getncattr(e) for m in measures] for e in ['long_name', 'units']})

    # create site variables
    sites = ['main_site']
    sites.extend(['exsites_{}'.format(e) for e in
                  paths.loc[paths.str.contains('exsites')].str.split('_').str[1].unique()])
    for s in sites:
        if s == 'main_site':
            idx = (~paths.str.contains('exsites'))
        else:
            idx = (paths.str.contains(s))

        for i, p in enumerate(paths.loc[idx]):
            temp_read = nc.Dataset(os.path.join(swg_dir, p))
            if i == 0:
                out_var = out.createVariable(s, float, ('day', 'w_var', 'real'), fill_value=np.nan)
                out_var.setncatts({
                    'long_name': 'site {}'.format(s),
                    'units': 'see nc variable w_var',
                    'lat': temp_read.lat,
                    'lon': temp_read.lon
                })

            else:
                out_var = out.variables[s]
            temp_out = np.zeros((days, len(measures))) * np.nan
            for j, m in enumerate(measures):
                temp_out[:, j] = np.array(temp_read.variables[m][:])

            out_var[:, :, i] = temp_out
    # set global variables including storyline, and yml
    out.description = 'amalgamated SWG run'
    out.created = datetime.datetime.now().isoformat()
    out.script = __file__
    out.storyline = str(storyline)
    out.yml = yml_txt
    out.sites = sites


measures = ['PR_A', 'Tmax', 'Tmin', 'RSDS', 'PEV']
measures_cor = ['rain', 'tmax', 'tmin', 'radn', 'pet']

rain_limits_wet = {
    # non wet months
    1: 40,
    2: 40,
    3: 40,
    4: 40,
    10: 40,
    11: 40,
    12: 40,

    # wet months
    5: 14,
    6: 11,
    7: 11,
    8: 13,
    9: 13,
}


def get_month_day_to_nonleap_doy(key_doy=False):
    """

    :param key_doy: bool, if true the keys are doy, else keys are (month, dayofmonth)
    :return: dictionary if not inverse: {(m,d}:doy} if inverse: {doy: (m,d)}
    """
    temp = pd.date_range('2025-01-01', '2025-12-31')  # a random non leap year
    day = temp.day
    month = temp.month
    doy = temp.dayofyear
    if key_doy:
        out = {dd: (m, d) for m, d, dd in zip(month, day, doy)}
    else:
        out = {(m, d): dd for m, d, dd in zip(month, day, doy)}

    return out


def get_monthly_smd_mean_detrended(leap=False, recalc=False):
    if leap:
        add = '366_cal'
    else:
        add = '365_cal'

    outpath = os.path.join(climate_shocks_env.supporting_data_dir, f'mean_montly_smd_detrend_{add}.csv')

    if not recalc and os.path.exists(outpath):
        average_smd = pd.read_csv(outpath, index_col=0)
        return average_smd.loc[:, 'smd'].to_dict()

    data = get_vcsn_record('detrended2').reset_index()
    if not leap:
        data = data.loc[~((data.date.dt.month == 2) & (data.date.dt.day == 29))]  # get rid of leap days
    average_start_year = 1981
    average_stop_year = 2010
    rain, pet, h2o_cap, h2o_start = data['rain'], data['pet'], 150, 1

    dates = data.loc[:, 'date']
    if leap:
        doy = dates.dt.dayofyear
    else:
        # reset doy to 365 calander (e.g. no leap effect)
        mapper = get_month_day_to_nonleap_doy(False)
        doy = [mapper[(m, d)] for m, d in zip(dates.dt.month, dates.dt.day)]
    pet = np.atleast_1d(pet)
    rain = np.atleast_1d(rain)

    assert dates.shape == pet.shape == rain.shape, 'date, pet, rain must be same shape'

    smd = calc_smd_monthly(rain, pet, dates,
                           month_start=detrended_start_month,
                           h2o_cap=150,
                           a=0.0073,
                           p=1, return_drn_aet=False)

    outdata = pd.DataFrame(data={'date': dates, 'doy': doy, 'pet': pet, 'rain': rain, 'smd': smd},
                           )

    # calculate mean smd for doy

    idx = (outdata.date.dt.year >= average_start_year) & (outdata.date.dt.year <= average_stop_year)
    temp = outdata.loc[idx, ['doy', 'smd']]
    average_smd = temp.groupby(doy).mean().set_index('doy')
    average_smd.to_csv(outpath)

    return average_smd.loc[:, 'smd'].to_dict()


if __name__ == '__main__':
    pass
