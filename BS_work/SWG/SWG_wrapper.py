"""
 Author: Matt Hanson
 Created: 10/02/2021 9:30 AM
 """
import os
import ksl_env
import numpy as np
import subprocess
import sys
import glob
import yaml
import pandas as pd
from Climate_Shocks.Stochastic_Weather_Generator.read_swg_data import read_swg_data
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_smd

oxford_lat, oxford_lon = -43.296, 172.192
swg = os.path.join(os.path.dirname(__file__), 'SWG_Final.py')


# note that each sim takes c. 0.12 mb of storage space.

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
    assert '_' not in os.path.basename(storyline_path), '_ in file basename mucks things up'
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


def run_SWG(yml_path, outdir, rm_npz=True, clean=True):
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
    if clean:
        clean_swg(outdir, yml_path)
    return '{}, {}'.format(result.stdout, result.stderr)


def clean_swg(swg_dir, yml_path, exsites=1):
    """
    remove swg files which do not match the data. Note that definitions are hard coded into this process
    via _check_data_v1.
    :param swg_dir: directory for the swg
    :param yml_path: path to the yml file
    :param exsites: int, number of external sites
    :return:
    """
    with open(yml_path, 'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_dict = yaml.load(file, Loader=yaml.FullLoader)
    storyline_path = param_dict['story_line_filename']
    storyline = pd.read_csv(storyline_path)
    paths = pd.Series(os.listdir(swg_dir))
    paths = paths.loc[(~paths.str.contains('exsites')) & (paths.str.contains('.nc'))]
    removed = []
    for p in paths:
        fp = os.path.join(swg_dir, p)
        bad = _check_data_v1(fp, storyline)
        if bad:
            removed.append(p)
            for i in range(exsites):
                vals = p.split('_')
                if len(vals) > 2:
                    raise ValueError('names preclude removal of exsites')
                os.remove(os.path.join(swg_dir, '{}exsites_P{}_{}'.format(vals[0], i, vals[1])))
            os.remove(fp)
    print('removed {} as they did not match the story: {}'.format(len(removed), '\n'.join(removed)))


def _check_data_v1(path, storyline):  # todo check

    storyline = storyline.set_index(['year', 'month'])
    data = read_swg_data(path)[0]

    # calc SMA
    data.loc[:, 'sma'] = calc_smd(rain=data.loc[:, 'rain'].values, pet=data.loc[:, 'pet'].values,
                                  h2o_cap=150, h2o_start=1, a=0.0073,
                                  p=1, return_drn_aet=False) - data.loc[:, 'doy'].replace(smd_mean)

    data.loc[:, 'wet'] = data.loc[:, 'rain'] >= 0.1
    data.loc[:, 'dry'] = data.loc[:, 'sma'] <= -20
    data.loc[:, 'hot'] = data.loc[:, 'tmax'] >= 25
    data.loc[:, 'cold'] = ((data.loc[:, 'tmin'] +
                            data.loc[:, 'tmax']) / 2).rolling(3).mean().fillna(method='bfill') <= 7

    temp = data.loc[:, ['year', 'month', 'wet',
                        'dry', 'hot', 'cold']].groupby(['year', 'month']).sum()
    storyline.loc[temp.index, ['wet', 'dry', 'hot', 'cold']] = temp.loc[:, ['wet', 'dry', 'hot', 'cold']]
    storyline.reset_index(inplace=True)

    storyline.loc[:, 'swg_precip_class'] = 'A'
    storyline.loc[((storyline.dry >= 10) &
                   np.in1d(storyline.month, [8, 9, 10, 11, 12, 1, 2, 3, 4, 5])), 'swg_precip_class'] = 'D'
    storyline.loc[((storyline.wet >= storyline.month.replace(rain_limits_wet)) &
                   np.in1d(storyline.month, [5, 6, 7, 8, 9])), 'swg_precip_class'] = 'W'

    storyline.loc[:, 'swg_temp_class'] = 'A'
    storyline.loc[(storyline.hot >= 7) & np.in1d(storyline.month, [11, 12, 1, 2, 3]), 'swg_temp_class'] = 'H'
    storyline.loc[(storyline.cold >= 10) & np.in1d(storyline.month, [5, 6, 7, 8, 9]), 'swg_temp_class'] = 'C'

    where_same = ((storyline.temp_class == storyline.swg_temp_class) & (
            storyline.precip_class == storyline.swg_precip_class))
    num_dif = (~((storyline.temp_class == storyline.swg_temp_class) & (
            storyline.precip_class == storyline.swg_precip_class))).sum()

    out_keys = ['{}:{}-{}_{}-{}'.format(m, p, swgp, t, swgt) for m, p, swgp, t, swgt in storyline.loc[~where_same, ['month',
                                                                                                   'precip_class',
                                                                                                   'swg_precip_class',
                                                                                                   'temp_class',
                                                                                                   'swg_temp_class'
                                                                                                   ]].itertuples(False,
                                                                                                                 None)]
    return num_dif, out_keys


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

# smd mean comes from line 133 (rounded) in Climate_Shocks/note_worthy_events/simple_soil_moisture_pet.py when called
# on line 76 of Climate_Shocks/note_worthy_events/inital_event_recurance.py
# this is the value set by the trended VCSN data averaged over the DOY.

smd_mean = {
    1: -79.0, 2: -79.0, 3: -81.0, 4: -82.0, 5: -82.0, 6: -84.0, 7: -85.0, 8: -84.0, 9: -84.0, 10: -83.0, 11: -82.0,
    12: -82.0, 13: -81.0, 14: -83.0, 15: -84.0, 16: -86.0, 17: -88.0, 18: -87.0, 19: -88.0, 20: -89.0, 21: -91.0,
    22: -91.0, 23: -91.0, 24: -91.0, 25: -90.0, 26: -91.0, 27: -91.0, 28: -92.0, 29: -93.0, 30: -95.0, 31: -95.0,
    32: -96.0, 33: -97.0, 34: -96.0, 35: -95.0, 36: -93.0, 37: -95.0, 38: -95.0, 39: -93.0, 40: -94.0, 41: -94.0,
    42: -91.0, 43: -89.0, 44: -89.0, 45: -85.0, 46: -85.0, 47: -85.0, 48: -84.0, 49: -84.0, 50: -83.0, 51: -82.0,
    52: -84.0, 53: -85.0, 54: -84.0, 55: -83.0, 56: -85.0, 57: -84.0, 58: -83.0, 59: -84.0, 60: -84.0, 61: -85.0,
    62: -82.0, 63: -83.0, 64: -83.0, 65: -83.0, 66: -83.0, 67: -83.0, 68: -84.0, 69: -84.0, 70: -82.0, 71: -81.0,
    72: -77.0, 73: -75.0, 74: -76.0, 75: -76.0, 76: -76.0, 77: -77.0, 78: -76.0, 79: -77.0, 80: -77.0, 81: -77.0,
    82: -75.0, 83: -75.0, 84: -74.0, 85: -75.0, 86: -75.0, 87: -75.0, 88: -74.0, 89: -73.0, 90: -72.0, 91: -72.0,
    92: -72.0, 93: -72.0, 94: -72.0, 95: -72.0, 96: -70.0, 97: -69.0, 98: -68.0, 99: -66.0, 100: -66.0, 101: -64.0,
    102: -64.0, 103: -64.0, 104: -64.0, 105: -63.0, 106: -64.0, 107: -64.0, 108: -64.0, 109: -63.0, 110: -63.0,
    111: -61.0, 112: -61.0, 113: -61.0, 114: -60.0, 115: -59.0, 116: -59.0, 117: -58.0, 118: -55.0, 119: -53.0,
    120: -50.0, 121: -49.0, 122: -47.0, 123: -48.0, 124: -47.0, 125: -46.0, 126: -46.0, 127: -45.0, 128: -45.0,
    129: -44.0, 130: -43.0, 131: -41.0, 132: -41.0, 133: -39.0, 134: -38.0, 135: -37.0, 136: -36.0, 137: -34.0,
    138: -33.0, 139: -32.0, 140: -29.0, 141: -28.0, 142: -27.0, 143: -26.0, 144: -25.0, 145: -23.0, 146: -22.0,
    147: -23.0, 148: -22.0, 149: -21.0, 150: -22.0, 151: -22.0, 152: -21.0, 153: -21.0, 154: -20.0, 155: -19.0,
    156: -19.0, 157: -19.0, 158: -18.0, 159: -17.0, 160: -17.0, 161: -17.0, 162: -17.0, 163: -16.0, 164: -16.0,
    165: -16.0, 166: -15.0, 167: -15.0, 168: -14.0, 169: -14.0, 170: -13.0, 171: -13.0, 172: -13.0, 173: -13.0,
    174: -13.0, 175: -12.0, 176: -12.0, 177: -12.0, 178: -11.0, 179: -11.0, 180: -10.0, 181: -9.0, 182: -9.0, 183: -8.0,
    184: -8.0, 185: -8.0, 186: -8.0, 187: -8.0, 188: -8.0, 189: -8.0, 190: -8.0, 191: -9.0, 192: -8.0, 193: -8.0,
    194: -7.0, 195: -7.0, 196: -8.0, 197: -6.0, 198: -5.0, 199: -5.0, 200: -4.0, 201: -3.0, 202: -3.0, 203: -3.0,
    204: -3.0, 205: -4.0, 206: -4.0, 207: -5.0, 208: -5.0, 209: -6.0, 210: -5.0, 211: -5.0, 212: -6.0, 213: -6.0,
    214: -6.0, 215: -6.0, 216: -7.0, 217: -7.0, 218: -7.0, 219: -7.0, 220: -7.0, 221: -8.0, 222: -8.0, 223: -7.0,
    224: -7.0, 225: -7.0, 226: -7.0, 227: -7.0, 228: -7.0, 229: -7.0, 230: -7.0, 231: -7.0, 232: -8.0, 233: -7.0,
    234: -8.0, 235: -8.0, 236: -9.0, 237: -8.0, 238: -8.0, 239: -9.0, 240: -9.0, 241: -9.0, 242: -9.0, 243: -10.0,
    244: -11.0, 245: -12.0, 246: -13.0, 247: -12.0, 248: -13.0, 249: -14.0, 250: -14.0, 251: -15.0, 252: -14.0,
    253: -15.0, 254: -16.0, 255: -17.0, 256: -18.0, 257: -18.0, 258: -20.0, 259: -21.0, 260: -22.0, 261: -23.0,
    262: -23.0, 263: -22.0, 264: -24.0, 265: -25.0, 266: -26.0, 267: -26.0, 268: -23.0, 269: -24.0, 270: -25.0,
    271: -25.0, 272: -25.0, 273: -27.0, 274: -28.0, 275: -29.0, 276: -27.0, 277: -27.0, 278: -27.0, 279: -28.0,
    280: -28.0, 281: -27.0, 282: -27.0, 283: -26.0, 284: -27.0, 285: -27.0, 286: -25.0, 287: -26.0, 288: -28.0,
    289: -29.0, 290: -29.0, 291: -31.0, 292: -33.0, 293: -35.0, 294: -36.0, 295: -38.0, 296: -38.0, 297: -39.0,
    298: -41.0, 299: -40.0, 300: -42.0, 301: -44.0, 302: -45.0, 303: -46.0, 304: -47.0, 305: -48.0, 306: -48.0,
    307: -49.0, 308: -51.0, 309: -51.0, 310: -52.0, 311: -52.0, 312: -53.0, 313: -55.0, 314: -58.0, 315: -59.0,
    316: -60.0, 317: -60.0, 318: -62.0, 319: -64.0, 320: -65.0, 321: -64.0, 322: -63.0, 323: -63.0, 324: -63.0,
    325: -59.0, 326: -61.0, 327: -62.0, 328: -62.0, 329: -60.0, 330: -60.0, 331: -61.0, 332: -62.0, 333: -61.0,
    334: -62.0, 335: -63.0, 336: -64.0, 337: -66.0, 338: -67.0, 339: -67.0, 340: -67.0, 341: -68.0, 342: -67.0,
    343: -68.0, 344: -69.0, 345: -70.0, 346: -70.0, 347: -71.0, 348: -72.0, 349: -73.0, 350: -73.0, 351: -73.0,
    352: -73.0, 353: -73.0, 354: -72.0, 355: -72.0, 356: -72.0, 357: -73.0, 358: -73.0, 359: -75.0, 360: -77.0,
    361: -79.0, 362: -78.0, 363: -78.0, 364: -78.0, 365: -78.0, 366: -83.0
}

if __name__ == '__main__':
    #todo triple check that these limits are not a problem!
    v7story = pd.read_csv(r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_csvs\test.csv')
    base0 = pd.read_csv(r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_csvs\0-baseline.csv')

    v7_paths = [
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S9.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S0.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S1.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S2.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S3.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S4.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S5.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S6.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S7.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\test\test_S8.nc",
    ]

    base0_paths = [
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S9.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S0.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S1.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S2.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S3.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S4.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S5.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S6.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S7.nc",
        r"D:\mh_unbacked\SLMACC_2020\SWG_runs\0-base\0-baseline_S8.nc",
    ]

    for p in base0_paths:
        print('test should pass', os.path.basename(p), _check_data_v1(p, base0))
        print('test should not pass', os.path.basename(p), _check_data_v1(p, v7story))
        print('\n')

    print('\n\n')
    for p in v7_paths:
        print('test should pass', os.path.basename(p), _check_data_v1(p, v7story))
        print('test should not pass', os.path.basename(p), _check_data_v1(p, base0))
        print('\n')
