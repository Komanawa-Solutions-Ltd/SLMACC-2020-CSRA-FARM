"""
one off testing script not a part of the processs!
 Author: Matt Hanson
 Created: 22/02/2021 9:42 AM
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


def check_single(path, yml_path, m=None):  # todo check

    if m == None:
        m = np.arange(1, 13)
    m = np.atleast_1d(m)

    with open(yml_path, 'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        param_dict = yaml.load(file, Loader=yaml.FullLoader)
    storyline_path = param_dict['story_line_filename']
    storyline = pd.read_csv(storyline_path)

    storyline = storyline.loc[np.in1d(storyline.month, m)]

    storyline = storyline.set_index(['year', 'month'])
    data = read_swg_data(path)[0]
    data = data.loc[np.in1d(data.month, m)]

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

    data.loc[:,'year'] += 1
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

    hot = storyline.hot.max()
    cold = storyline.cold.max()
    wet = storyline.wet.max()
    dry = storyline.dry.max()

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
    return num_dif, out_keys ,hot, cold, wet, dry


if __name__ == '__main__':

    base_dir = os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', 'try_individual')
    out_dict = {}
    for d in os.listdir(base_dir):
        if '.csv' in d:
            continue
        print(d)
        outdir = os.path.join(base_dir, d)
        yml_path = os.path.join(outdir, 'ind.yml')
        paths = pd.Series(os.listdir(outdir))
        paths = paths.loc[(~paths.str.contains('exsites')) & (paths.str.contains('.nc'))]
        outdata = pd.DataFrame(index=paths, columns=['count'])
        for p in paths:
            num_dif, out_keys, hot, cold, wet, dry = check_single(os.path.join(outdir, p), yml_path, m=int(p.split('-')[0].replace('m', '')))
            outdata.loc[p, 'count'] = num_dif
            for k in ['hot', 'cold', 'wet', 'dry']:
                outdata.loc[p, k] = eval(k)
        outdata.to_csv(os.path.join(outdir, 'num_diff.csv'))
        out_dict[d] = outdata.loc[:,'count'].mean()
    pd.Series(out_dict).to_csv(os.path.join(base_dir, 'event_overview.csv'))

