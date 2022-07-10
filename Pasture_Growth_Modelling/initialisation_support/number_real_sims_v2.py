"""
created matt_dumont 
on: 11/07/22
"""
import netCDF4 as nc
from pathlib import Path
import os
import ksl_env
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob


def make_comparison(path):
    ns_to_compair = [1, 10, 100, 500]
    n = 100
    outdata = []
    path = Path(path)
    dataset = nc.Dataset(path)
    all_pg = np.array(dataset.variables['m_PGR'])
    # todo get just the unique event
    months = np.array(dataset.variables['m_month'])
    idx = np.where(months == int(path.name.split('-')[0].replace('m', '')))[0][0]
    all_pg = all_pg[idx]
    check_month = months[idx]
    true_val = all_pg.mean()
    for ncomp in ns_to_compair:
        temp = np.random.choice(all_pg, (n, ncomp))
        outdata.append(temp.mean(axis=1))
    outdata = np.array(outdata)
    return true_val, outdata


def plot_monthly(base_path, m):
    paths = Path(base_path).glob(f'm{m:02}*eyrewell-irrigated.nc')
    data, name = [], []
    single_data = []
    for p in paths:
        name.append(p.name)
        true_val, outdata = make_comparison(p)
        outdata = outdata/true_val
        data.append(outdata)
        single_data.append(outdata[0])


    pass


if __name__ == '__main__':
    temp_data = '/home/matt_dumont/Downloads/m01-A-A-50-eyrewell-irrigated.nc'
    make_comparison(temp_data)
    pass
    # base_path = Path("D:\mh_unbacked\SLMACC_2020_norm\pasture_growth_sims\unique_event_events")
