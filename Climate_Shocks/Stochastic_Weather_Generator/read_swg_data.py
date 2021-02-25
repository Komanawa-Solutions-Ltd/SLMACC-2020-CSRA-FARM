"""
 Author: Matt Hanson
 Created: 12/02/2021 2:28 PM
 """

import netCDF4 as nc
import pandas as pd
import numpy as np


def change_swg_units(data):
    for k in data.keys():
        if k in ['tmax', 'tmin', 'Tmax', 'Tmin']:
            # convert to C.
            # originally in k
            data.loc[:, k] += - 273.15
        elif k in ['pet', 'rain', 'PR_A', 'PEV']:
            # convert to mm
            # orginally in kg/m2/s or mm/m2/s
            # kg/m2 ==mm/m2
            data.loc[:, k] *= 86400
        elif k in ['radn', 'RSDS']:
            # convert to MJ/m2/d
            # orignally in W/m2
            data.loc[:, k] *= 86400 * 1e-6
        else:
            continue


def read_swg_data(paths):
    """
    read teh data and put it in the correct format for BASGRA
    :param paths: list of paths to indidivual netcdf files
    :return:
    """

    # weather data with datetime index named date, no missing days, and has at least the following
    #                          keys: ['year', 'doy', 'radn', 'tmin', 'tmax', 'rain', 'pet']
    paths = np.atleast_1d(paths)
    outdata = []
    for p in paths:
        temp = nc.Dataset(p)
        start_month = int(np.array(temp.variables['Month'][0]))
        sim_len = temp.dimensions['day'].size
        idx = pd.date_range('2025-{:02d}-01'.format(start_month), periods=sim_len, freq='D')
        temp_out = pd.DataFrame(index=idx, columns=['year', 'doy', 'radn', 'tmin', 'tmax', 'rain', 'pet'])
        temp_out.loc[:, 'radn'] = np.array(temp.variables['RSDS'])
        temp_out.loc[:, 'tmin'] = np.array(temp.variables['Tmin'])
        temp_out.loc[:, 'tmax'] = np.array(temp.variables['Tmax'])
        temp_out.loc[:, 'rain'] = np.array(temp.variables['PR_A'])
        temp_out.loc[:, 'pet'] = np.array(temp.variables['PEV'])
        temp_out.loc[:, 'doy'] = temp_out.index.dayofyear
        temp_out.loc[:, 'year'] = temp_out.index.year
        temp_out.loc[:, 'month'] = temp_out.index.month
        temp_out.index.name = 'date'
        change_swg_units(temp_out)
        outdata.append(temp_out)
        temp.close()

    return outdata


if __name__ == '__main__':
    read_swg_data([r"C:\Users\Matt Hanson\Downloads\10_rm_npz\v7exsites_P0_S0.nc"])
