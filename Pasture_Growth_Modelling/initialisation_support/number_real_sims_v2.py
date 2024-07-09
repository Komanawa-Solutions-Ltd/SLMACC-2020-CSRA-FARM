"""
created matt_dumont 
on: 11/07/22
"""
import netCDF4 as nc
from pathlib import Path
import os
import project_base
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

month_to_month = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec',

}


def make_comparison(path):
    ns_to_compair = [1, 10, 100, 500]
    n = 1000
    outdata = []
    path = Path(path)
    dataset = nc.Dataset(path)
    all_pg = np.array(dataset.variables['m_PGR'])
    months = np.array(dataset.variables['m_month'])
    idx = np.where(months == int(path.name.split('-')[0].replace('m', '')))[0][0]
    all_pg = all_pg[idx]
    check_month = months[idx]
    true_val = all_pg.mean()
    for ncomp in ns_to_compair:
        temp = np.random.choice(all_pg, (n, ncomp))
        outdata.append(temp.mean(axis=1))
    outdata = np.array(outdata)
    return true_val, outdata, ns_to_compair


def plot_monthly(base_path, m, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    paths = Path(base_path).glob(f'm{m:02}*eyrewell-irrigated.nc')
    data, name = [], []
    single_data = []
    for p in paths:
        name.append(p.name.split('-eyrewell-irrigated')[0].replace(f'm{m:02}-', ''))
        true_val, outdata, ns_to_compair = make_comparison(p)
        outdata = (outdata - true_val) / true_val
        data.append(outdata)
        single_data.append(outdata[0])

    # plot single data
    fig, axs = plt.subplots(5, 1, sharey=True, figsize=(10, 8))
    all_axs = []
    all_axs.extend(axs)
    start = 0
    print(len(single_data))
    for i, ax in enumerate(all_axs):
        if i == 4:
            stop = len(single_data)
        else:
            stop = start + 10
        if start >= len(single_data):
            continue
        temp = single_data[start:stop]
        temp_names = name[start:stop]
        ax.boxplot(temp, labels=temp_names)
        start = stop
    fig.suptitle(f'Single Iterations for the unique events of {month_to_month[m]}')
    fig.supylabel('Relative difference to mean')
    fig.tight_layout()
    fig.savefig(outdir.joinpath(f'{m}_single_data.png'))

    # show min and max stdv
    stv = []
    for s in single_data:
        stv.append(np.std(s))
    stv = np.array(stv)
    maxidx = np.argmax(stv)
    minidx = np.argmin(stv)
    t = np.abs(stv - stv.mean())
    mid_idx = np.argmin(t)
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 8))
    for i, (ax, idx, n) in enumerate(zip(axs, [minidx, mid_idx, maxidx], ['minimum', 'middle', 'maximum'])):
        temp = data[idx]
        temp_name = name[idx]
        temp = [e for e in temp]
        ax.boxplot(temp, labels=ns_to_compair)
        ax.set_title(f'{n.capitalize()} standard deviation ({temp_name})')
    fig.supylabel('Relative difference to mean')
    fig.supxlabel('Number of realisations')
    fig.tight_layout()

    fig.savefig(outdir.joinpath(f'{m}_realisations_data.png'))
    pass


if __name__ == '__main__':
    base_path = Path(r"D:\mh_unbacked\SLMACC_2020_norm\pasture_growth_sims\unique_event_events")
    outdir = r"M:\Shared drives\Z2003_SLMACC\0_Y2_and_Final_Reporting\final_plots\n_realisations"
    for m in range(1,13):
        print(m)
        plot_monthly(base_path, m, outdir)
    pass
