"""
 Author: Matt Hanson
 Created: 26/02/2021 12:44 PM
 """

import os
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from Pasture_Growth_Modelling.full_model_implementation import out_variables, out_metadata, month_len
from Pasture_Growth_Modelling.historical_average_baseline import get_historical_average_baseline

default_outvars = [e for e in out_variables] + ['PGRA', 'PGRA_cum']


def plot_sims(data_paths, plot_ind=False, plt_vars=default_outvars, nindv=100, save_dir=None, show=False,
              figsize=(11, 8),
              daily=False, ex_save='', plot_baseline=True, site=None, mode=None):  # todo pass through site,mode
    """
    plot multiple basgra netcdf simulations, one plot per variable, with all datasets on the same plot
    :param data_paths: paths to the netcdf files
    :param plot_ind: boolean if True then plot individual simulations in additino to the mean (these are dash-dot and
                     aplha = 0.5)
    :param plt_vars: the basgra variables from the netcdf to plot
    :param nindv: the number of individual simulations to plot
    :param save_dir: none or a directory to save plots, if None do not plot
    :param show: bool if True call plt.show at the end
    :param figsize: figure size passed directly to plt.subplots
    :param daily: boolean, if True plot daily data, if False plot monthly data, note that daily data may not be
                  avalible for all sims
    :param ex_save: additional str to add to end of file name
    :return:
    """
    assert np.in1d(plt_vars, default_outvars).all(), (f'some variables {plt_vars} are not found in the '
                                                      f'expected variables: {out_variables}')
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    axs = {}
    figs = {}
    for v in plt_vars:
        figs[v], axs[v] = fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(v)
        ax.set_xlabel('date')
        ax.set_ylabel(f'{v} ({out_metadata[v]["unit"]})')

    cmap = get_cmap('tab20')
    n_scens = len(data_paths)
    colors = [cmap(e / n_scens) for e in range(n_scens)]  # pick from color map

    for p, c in zip(data_paths, colors):
        temp = nc.Dataset(p)
        pname = os.path.splitext(os.path.basename(p))[0]

        if daily:
            app = 'd'
            freq = 'D'
        else:
            app = 'm'
            freq = 'MS'
        sm = np.array(temp.variables['m_month'])[0]
        em = np.array(temp.variables['m_month'])[-1]
        sy = np.array(temp.variables['m_year'])[0]
        ey = np.array(temp.variables['m_year'])[-1]
        x = pd.date_range(f'{sy}-{sm:02d}-1', f'{ey}-{em:02d}-{month_len[em]}', freq=freq)

        if plot_ind:
            idxs = np.random.randint(temp.dimensions['realisation'].size, size=(nindv,))
            for v in plt_vars:
                fix, ax = figs[v], axs[v]
                data = np.array(temp.variables[f'{app}_{v}'][:, idxs])
                for i in range(data.shape[-1]):
                    ax.plot(x, data[:, i], c=c, alpha=0.3, ls=':')
        for v in plt_vars:
            data = np.array(temp.variables[f'{app}_{v}'])
            fix, ax = figs[v], axs[v]
            ax.plot(x, np.nanmean(data, axis=1), c=c, label=f'mean {pname}', linewidth=3, marker='o')
            if plot_baseline:
                base, run_date = get_historical_average_baseline(site, mode, years=x.year.unique(), key=v)
                base = base.set_index(['year', 'month']).drop(columns='doy')
                base = base.loc[zip(x.year, x.month)].reset_index().drop_duplicates().loc[:, v]
                # todo get temp the right length
                ax.plot(x, base, c='grey', label=f'historical mean {pname}', linewidth=3, marker='o')

    for v in plt_vars:
        fig, ax = figs[v], axs[v]
        ax.set_title(v)
        ax.set_xlabel('date')
        ax.set_ylabel(f'{v} ({out_metadata[v]["unit"]})')
        ax.legend()
    if save_dir is not None:
        for v, fig in figs.items():
            fig.savefig(os.path.join(save_dir, f'{v}{ex_save}.png'))
            if not show:
                plt.close(fig)
    if show:
        plt.show()


if __name__ == '__main__':
    plot_sims([
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\test_pg_ex_swg\0-baseline-paddock_0-eyrewell-irrigated.nc",
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\test_pg_ex_swg\0-baseline-paddock_1-eyrewell-irrigated.nc",
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\test_pg_ex_swg\0-baseline-paddock_2-eyrewell-irrigated.nc",
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\test_pg_ex_swg\0-baseline-paddock_3-eyrewell-irrigated.nc",

    ], plot_ind=False, plt_vars=out_variables, nindv=30, save_dir=None, show=True,
        daily=False)

    plot_sims([
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\test_pg_ex_swg\0-baseline-paddock-mean-eyrewell-irrigated.nc",
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\test_pg_ex_swg\0-baseline-eyrewell-irrigated.nc",

    ], plot_ind=False, plt_vars=out_variables, nindv=30, save_dir=None, show=True,
        daily=False)
