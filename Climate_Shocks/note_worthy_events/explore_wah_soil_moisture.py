"""
this file is for loose exploration, the datasets are not included in the files stream.
 Author: Matt Hanson
 Created: 6/01/2021 10:13 AM
 """
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import glob
import os

lat, lon = -43.372, 172.333


def get_sim(sim_path, lat=lat, lon=lon):
    variables = [
        'pstar',
        'sw_flux',
        'wind_speed',
        'precipitation',
        'mslp',
        'rh_max',
        'tmax',
        'dewpoint',
        'rh_min',
        'spec_h',
        'wind_u',
        'tmin',
        'wind_v',
        'soil_moisture',
    ]
    data = nc.Dataset(sim_path)
    dates = np.array(nc.num2date(data.variables['time0'], data.variables['time0'].units))
    dates = pd.to_datetime([e._to_real_datetime() for e in dates])
    outdata = pd.DataFrame(index=dates, columns=variables)
    temp = (np.abs(data.variables['global_latitude0'][:] - lat) ** 2 +
            np.abs(data.variables['global_longitude0'][:] - lon) ** 2)
    lat_idx, lon_idx = np.where(temp == temp.min())
    lat_idx = lat_idx[0]
    lon_idx = lon_idx[0]
    print(
        'use_coords:',
        data.variables['global_latitude0'][lat_idx, lon_idx],
        data.variables['global_longitude0'][lat_idx, lon_idx],
        'lat_idx', lat_idx, 'lon_idx', lon_idx
    )
    for v in variables:
        outdata.loc[:, v] = np.array(data.variables[v][:, 0, lat_idx, lon_idx])
    outdata.loc[:, 'precipitation'] *= 86400
    return outdata


def plot_sims(sim_paths, vars, lat=lat, lon=lon):
    sim_paths = np.atleast_1d(sim_paths)
    fig, axs = plt.subplots(nrows=len(vars), sharex=True)
    n_scens = len(sim_paths)
    cmap = get_cmap('tab20')
    colors = [cmap(e / n_scens) for e in range(n_scens)]  # pick from color map
    for c, sim in zip(colors, sim_paths):
        data = get_sim(sim)
        for ax, v in zip(axs, vars):
            ax.plot(data.index, data[v], c=c)
            ax.set_ylabel(v)


if __name__ == '__main__':
    paths = [
        r"C:\Users\Matt Hanson\Downloads\w@h for matt-20210105T205116Z-001\w@h for matt\wah2_anz50_a0d7_201612_20_793_011761121_2.nc",
        r"C:\Users\Matt Hanson\Downloads\w@h for matt-20210105T205116Z-001\w@h for matt\wah2_anz50_a0d9_201612_20_793_011761123_0.nc",
    ]
    paths = glob.glob(
        r"C:\Users\Matt Hanson\Downloads\w@h for matt-20210105T205116Z-001\w@h for matt\*.nc"
    )
    vars = [
        # 'wind_speed',
        # 'rh_max',
        # 'tmax',
        'precipitation',
        'soil_moisture',

    ]
    plot_sims(paths, vars)
    plt.show()  # todo start working through this!
    # inital conditions will be important to think through, inital guess if we chop off the first month we
    # should be ok... best to use just the internal water year starting 2017-07-01
    # todo look through many months and get a sense of the daily distributions for SMA
    # todo kinda looks like it does not have runoff..., or similar.
    # check soil moisture units, I think they are kg/m2 or mm, yep
