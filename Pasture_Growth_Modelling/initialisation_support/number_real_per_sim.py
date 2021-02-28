"""
 Author: Matt Hanson
 Created: 16/02/2021 11:14 AM
 """
import netCDF4 as nc
import os
import ksl_env
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

default_funcs = (
    #'np.nanmedian',
    'np.nanmean',
)


def make_dataset(all_data, n, n_compare,
                 functions=default_funcs):  # todo looks like we can get away with 100-1000 sims, but check on other storylines!
    """

    :param all_data: nc dataset
    :param n: number of realisations to sample
    :param n_compare: number of resampling sets to compare
    :param functions: strings of functions to run
    :return:
    """
    options = np.array(all_data.variables['real'])
    pass
    all_out = {}
    for k in functions:
        outdata = np.zeros((n_compare, all_data.dimensions['sim_month'].size)) * np.nan
        all_pg = np.array(all_data.variables['m_PGR']).transpose()
        act_mean = eval(k)(all_pg, axis=0)
        for i in range(n_compare):
            idx = np.random.choice(options, n)
            outdata[i] = eval(k)(all_pg[idx], axis=0)
        difs = outdata - act_mean[np.newaxis, :]
        all_out[k] = (act_mean, outdata, difs)
    return all_out


def plot_resampled_sims(paths, ns, n_compare, show=True, save_dir=None):
    for p in paths:
        all_data = nc.Dataset(p)
        x = pd.date_range('2025-07-01', freq='MS', periods=all_data.dimensions['sim_month'].size)
        ns = np.atleast_1d(ns)
        for f in default_funcs:
            outdata_all = []
            for n in ns:
                data = make_dataset(all_data, n, n_compare)
                act_mean, outdata, difs = data[f]
                outdata_all.append(np.nanmax(np.abs(difs), axis=1))
                fig, (ax, ax2) = plt.subplots(2, figsize=(11, 11))

                for i in range(n_compare):
                    ax.plot(x, outdata[i], c='r', alpha=0.5, ls='--')
                ax.plot(x, act_mean, c='b')
                ax.set_ylim(0, 120)
                ax.set_title('{} - {}:\n{} vs 10000'.format(f, os.path.basename(p), n))
                ax.set_ylabel('PGR')
                ax.set_xlabel('PGR dif')
                ax2.boxplot(difs, labels=x)
                if save_dir is not None:
                    use_save_dir = os.path.join(save_dir,os.path.basename(p).split('.')[0])
                    if not os.path.exists(use_save_dir):
                        os.makedirs(use_save_dir)
                    fig.savefig(os.path.join(use_save_dir, '{}_{}_{}_comp.png'.format(os.path.basename(p).split('.')[0],
                                                                                  f, n)))
                if not show:
                    plt.close(fig)
            fig, ax = plt.subplots(figsize=(11, 11))
            ax.boxplot(outdata_all, labels=ns)
            ax.set_yscale('log')
            ax.set_title('{} - {}:\n n vs 10000'.format(f, os.path.basename(p), ))
            ax.set_xlabel('number of sims per average')
            ax.set_ylabel('maximum error from 10,000 simulation {}'.format(f))
            ax.set_ylim(0.1, 100)
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                fig.savefig(os.path.join(save_dir, 'z-max_diff_{}_{}.png'.format(os.path.basename(p).split('.')[0],
                                                                                 f)))

    if show:
        plt.show()


if __name__ == '__main__':
    # todo re look at given new generation technique
    all_paths = [  # todo add irrigated base and keep looking at this!
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-oxford-dryland.nc",
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-oxford-irrigated.nc",
        r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\baseline_sim\0-baseline-eyrewell-irrigated.nc",

    ]
    plot_resampled_sims(all_paths,
                        [1, 10, 100, 200, 300, 400, 500, 750, 1000, 2500, 5000, 7500], 1000,
                        save_dir=os.path.join(ksl_env.slmmac_dir_unbacked,
                                              'pasture_growth_sims', 'n_comp_plots'),
                        show=False)
