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
from Climate_Shocks.get_past_record import get_vcsn_record

oxford_lat, oxford_lon = -43.296, 172.192
swg = os.path.join(os.path.dirname(__file__), 'SWG_Final.py')
default_vcf = os.path.join(os.path.dirname(__file__), 'event_definition_data_fixed.csv')


# note that each sim takes c. 0.12 mb of storage space.

def create_yaml(outpath_yml, outsim_dir, nsims, storyline_path, sim_name=None,
                base_dir=os.path.dirname(ksl_env.get_vscn_dir()),
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
                                  p=1, return_drn_aet=False) - data.loc[:, 'doy'].replace(
        smd_mean_detrended)  # todo trended or de trended!

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

    out_keys = ['{}:{}-{}_{}-{}'.format(m, p, swgp, t, swgt) for m, p, swgp, t, swgt in
                storyline.loc[~where_same, ['month',
                                            'precip_class',
                                            'swg_precip_class',
                                            'temp_class',
                                            'swg_temp_class'
                                            ]].itertuples(False,
                                                          None)]
    return num_dif, out_keys


def _check_data_v2(path, storyline):  # todo check
    tolerance = 2

    storyline = storyline.set_index(['year', 'month'])
    data = read_swg_data(path)[0]

    # calc SMA
    data.loc[:, 'sma'] = calc_smd(rain=data.loc[:, 'rain'].values, pet=data.loc[:, 'pet'].values,
                                  h2o_cap=150, h2o_start=1, a=0.0073,
                                  p=1, return_drn_aet=False) - data.loc[:, 'doy'].replace(
        smd_mean_detrended)  # todo trended or de trended!

    data.loc[:, 'wet'] = data.loc[:, 'rain'] >= 0.1
    data.loc[:, 'dry'] = data.loc[:, 'sma'] <= -20
    data.loc[:, 'hot'] = data.loc[:, 'tmax'] >= 25
    data.loc[:, 'cold'] = ((data.loc[:, 'tmin'] +
                            data.loc[:, 'tmax']) / 2).rolling(3).mean().fillna(method='bfill') <= 7

    temp = data.loc[:, ['year', 'month', 'wet',
                        'dry', 'hot', 'cold']].groupby(['year', 'month']).sum()
    storyline.loc[temp.index, ['wet', 'dry', 'hot', 'cold']] = temp.loc[:, ['wet', 'dry', 'hot', 'cold']]
    storyline.reset_index(inplace=True)

    for i, m, p, t, w, d, h, c in storyline.loc[:, ['month', 'precip_class',
                                                    'temp_class', 'wet',
                                                    'dry', 'hot', 'cold']].itertuples(True, None):
        if p == 'A':
            storyline.loc[i, 'precip_match'] = (d < (10 + tolerance)) and (w < (rain_limits_wet[m] + tolerance))
        elif p == 'D':
            storyline.loc[i, 'precip_match'] = (d >= (10 - tolerance))
        elif p == 'W':
            storyline.loc[i, 'precip_match'] = (w >= (rain_limits_wet[m] - tolerance))
        else:
            raise ValueError('shouldnt get here')

        if t == 'A':
            storyline.loc[i, 'temp_match'] = (h < (7 + tolerance)) and (c < (10 + tolerance))
        elif t == 'H':
            storyline.loc[i, 'temp_match'] = (h >= (7 - tolerance))
        elif t == 'C':
            storyline.loc[i, 'temp_match'] = (c >= (10 - tolerance))
        else:
            raise ValueError('shouldnt get here')

    storyline.loc[:, 'swg_precip_class'] = 'A'
    storyline.loc[((storyline.dry >= 10) &
                   np.in1d(storyline.month, [8, 9, 10, 11, 12, 1, 2, 3, 4, 5])), 'swg_precip_class'] = 'D'
    storyline.loc[((storyline.wet >= storyline.month.replace(rain_limits_wet)) &
                   np.in1d(storyline.month, [5, 6, 7, 8, 9])), 'swg_precip_class'] = 'W'

    storyline.loc[:, 'swg_temp_class'] = 'A'
    storyline.loc[(storyline.hot >= 7) & np.in1d(storyline.month, [11, 12, 1, 2, 3]), 'swg_temp_class'] = 'H'
    storyline.loc[(storyline.cold >= 10) & np.in1d(storyline.month, [5, 6, 7, 8, 9]), 'swg_temp_class'] = 'C'
    where_same = (storyline.temp_match & storyline.precip_match)
    num_dif = (~(storyline.temp_match & storyline.precip_match)).sum()

    out_keys = ['{}:{}-{}_{}-{}'.format(m, p, swgp, t, swgt) for m, p, swgp, t, swgt in
                storyline.loc[~where_same, ['month',
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

smd_mean_trended = {
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

# made from the detrended SMD data, made in the function below
smd_mean_detrended = {
    1: -86.0, 2: -86.0, 3: -88.0, 4: -89.0, 5: -89.0, 6: -91.0, 7: -92.0, 8: -91.0, 9: -91.0, 10: -91.0, 11: -89.0,
    12: -89.0, 13: -88.0, 14: -90.0, 15: -91.0, 16: -92.0, 17: -94.0, 18: -94.0, 19: -95.0, 20: -96.0, 21: -98.0,
    22: -98.0, 23: -98.0, 24: -98.0, 25: -97.0, 26: -98.0, 27: -98.0, 28: -98.0, 29: -100.0, 30: -101.0, 31: -101.0,
    32: -102.0, 33: -103.0, 34: -103.0, 35: -101.0, 36: -100.0, 37: -101.0, 38: -101.0, 39: -100.0, 40: -101.0,
    41: -100.0, 42: -97.0, 43: -96.0, 44: -95.0, 45: -92.0, 46: -92.0, 47: -92.0, 48: -91.0, 49: -90.0, 50: -90.0,
    51: -90.0, 52: -91.0, 53: -92.0, 54: -91.0, 55: -90.0, 56: -92.0, 57: -91.0, 58: -90.0, 59: -91.0, 60: -91.0,
    61: -92.0, 62: -89.0, 63: -90.0, 64: -90.0, 65: -90.0, 66: -90.0, 67: -90.0, 68: -91.0, 69: -91.0, 70: -90.0,
    71: -88.0, 72: -84.0, 73: -82.0, 74: -83.0, 75: -83.0, 76: -83.0, 77: -83.0, 78: -83.0, 79: -84.0, 80: -84.0,
    81: -84.0, 82: -83.0, 83: -82.0, 84: -82.0, 85: -82.0, 86: -83.0, 87: -82.0, 88: -81.0, 89: -81.0, 90: -80.0,
    91: -80.0, 92: -80.0, 93: -80.0, 94: -80.0, 95: -80.0, 96: -78.0, 97: -77.0, 98: -76.0, 99: -75.0, 100: -74.0,
    101: -73.0, 102: -73.0, 103: -73.0, 104: -73.0, 105: -73.0, 106: -73.0, 107: -74.0, 108: -74.0, 109: -73.0,
    110: -72.0, 111: -71.0, 112: -71.0, 113: -71.0, 114: -70.0, 115: -69.0, 116: -69.0, 117: -68.0, 118: -66.0,
    119: -63.0, 120: -61.0, 121: -60.0, 122: -59.0, 123: -59.0, 124: -59.0, 125: -57.0, 126: -57.0, 127: -57.0,
    128: -56.0, 129: -55.0, 130: -54.0, 131: -53.0, 132: -53.0, 133: -51.0, 134: -50.0, 135: -49.0, 136: -47.0,
    137: -46.0, 138: -44.0, 139: -42.0, 140: -40.0, 141: -39.0, 142: -38.0, 143: -37.0, 144: -35.0, 145: -33.0,
    146: -33.0, 147: -33.0, 148: -32.0, 149: -32.0, 150: -32.0, 151: -32.0, 152: -32.0, 153: -31.0, 154: -30.0,
    155: -29.0, 156: -29.0, 157: -29.0, 158: -28.0, 159: -28.0, 160: -27.0, 161: -27.0, 162: -26.0, 163: -25.0,
    164: -24.0, 165: -24.0, 166: -23.0, 167: -23.0, 168: -22.0, 169: -22.0, 170: -21.0, 171: -21.0, 172: -21.0,
    173: -21.0, 174: -21.0, 175: -20.0, 176: -20.0, 177: -20.0, 178: -20.0, 179: -19.0, 180: -18.0, 181: -17.0,
    182: -16.0, 183: -15.0, 184: -16.0, 185: -16.0, 186: -15.0, 187: -14.0, 188: -14.0, 189: -14.0, 190: -14.0,
    191: -14.0, 192: -14.0, 193: -13.0, 194: -13.0, 195: -13.0, 196: -13.0, 197: -12.0, 198: -11.0, 199: -9.0,
    200: -8.0, 201: -7.0, 202: -7.0, 203: -7.0, 204: -7.0, 205: -8.0, 206: -8.0, 207: -9.0, 208: -10.0, 209: -10.0,
    210: -9.0, 211: -10.0, 212: -10.0, 213: -11.0, 214: -10.0, 215: -11.0, 216: -11.0, 217: -11.0, 218: -12.0,
    219: -11.0, 220: -12.0, 221: -12.0, 222: -13.0, 223: -11.0, 224: -11.0, 225: -11.0, 226: -11.0, 227: -12.0,
    228: -12.0, 229: -12.0, 230: -12.0, 231: -11.0, 232: -12.0, 233: -12.0, 234: -12.0, 235: -13.0, 236: -13.0,
    237: -12.0, 238: -12.0, 239: -13.0, 240: -13.0, 241: -13.0, 242: -14.0, 243: -15.0, 244: -16.0, 245: -17.0,
    246: -18.0, 247: -17.0, 248: -18.0, 249: -19.0, 250: -20.0, 251: -20.0, 252: -20.0, 253: -20.0, 254: -22.0,
    255: -23.0, 256: -23.0, 257: -24.0, 258: -25.0, 259: -27.0, 260: -28.0, 261: -29.0, 262: -29.0, 263: -28.0,
    264: -30.0, 265: -31.0, 266: -32.0, 267: -31.0, 268: -29.0, 269: -29.0, 270: -31.0, 271: -30.0, 272: -31.0,
    273: -32.0, 274: -34.0, 275: -35.0, 276: -34.0, 277: -33.0, 278: -34.0, 279: -35.0, 280: -35.0, 281: -34.0,
    282: -34.0, 283: -33.0, 284: -34.0, 285: -33.0, 286: -32.0, 287: -33.0, 288: -35.0, 289: -36.0, 290: -37.0,
    291: -39.0, 292: -40.0, 293: -42.0, 294: -44.0, 295: -46.0, 296: -46.0, 297: -47.0, 298: -49.0, 299: -48.0,
    300: -50.0, 301: -52.0, 302: -53.0, 303: -54.0, 304: -55.0, 305: -57.0, 306: -56.0, 307: -58.0, 308: -59.0,
    309: -59.0, 310: -60.0, 311: -60.0, 312: -62.0, 313: -64.0, 314: -66.0, 315: -67.0, 316: -68.0, 317: -69.0,
    318: -70.0, 319: -72.0, 320: -74.0, 321: -72.0, 322: -71.0, 323: -71.0, 324: -71.0, 325: -68.0, 326: -69.0,
    327: -70.0, 328: -70.0, 329: -69.0, 330: -69.0, 331: -70.0, 332: -71.0, 333: -70.0, 334: -71.0, 335: -73.0,
    336: -73.0, 337: -74.0, 338: -75.0, 339: -76.0, 340: -75.0, 341: -77.0, 342: -76.0, 343: -77.0, 344: -78.0,
    345: -78.0, 346: -78.0, 347: -79.0, 348: -80.0, 349: -81.0, 350: -81.0, 351: -81.0, 352: -81.0, 353: -82.0,
    354: -80.0, 355: -80.0, 356: -80.0, 357: -81.0, 358: -81.0, 359: -82.0, 360: -84.0, 361: -86.0, 362: -85.0,
    363: -86.0, 364: -85.0, 365: -86.0, 366: -92.0
}


def make_smd_mean_detrended():
    data = get_vcsn_record('detrended2').reset_index()
    average_start_year = 1981
    average_stop_year = 2010
    rain, pet, date, h2o_cap, h2o_start = data['rain'], data['pet'], data.date, 150, 1

    date = np.atleast_1d(date)
    doy = pd.Series(date).dt.dayofyear

    pet = np.atleast_1d(pet)
    rain = np.atleast_1d(rain)

    assert date.shape == pet.shape == rain.shape, 'date, pet, rain must be same shape'

    smd, drain, aet_out = calc_smd(rain, pet, h2o_cap, h2o_start, a=0.0073, p=1, return_drn_aet=True)

    outdata = pd.DataFrame(data={'date': date, 'doy': doy, 'pet': pet, 'rain': rain, 'smd': smd, 'drain': drain,
                                 'aet_out': aet_out},
                           )

    # calculate mean smd for doy

    idx = (outdata.date.dt.year >= average_start_year) & (outdata.date.dt.year <= average_stop_year)
    temp = outdata.loc[idx, ['doy', 'smd']]
    average_smd = temp.groupby(doy).mean().set_index('doy')
    pass


if __name__ == '__main__':
    make_smd_mean_detrended()
    # todo triple check that these limits are not a problem!
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
