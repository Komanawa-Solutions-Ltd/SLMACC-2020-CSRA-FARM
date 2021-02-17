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

#todo run on baseline scen, and look at mean vs median.

def make_dataset(n, n_compare): #todo looks like we can get away with 100 sims, but check on other storylines!
    all_data = nc.Dataset(os.path.join(ksl_env.slmmac_dir_unbacked,
                                       'test_full_model0000',
                                       'test10000-eyrewell-irrigated.nc'))
    options = np.array(all_data.variables['real'])
    pass
    outdata = np.zeros((n_compare, all_data.dimensions['sim_month'].size)) * np.nan
    all_pg = np.array(all_data.variables['m_PGR']).transpose()
    act_mean = np.nanmean(all_pg, axis=0)
    for i in range(n_compare):
        idx = np.random.choice(options, n)
        outdata[i] = np.nanmean(all_pg[idx], axis=0)
    difs = outdata - act_mean[np.newaxis, :]
    return act_mean, outdata, difs


def plot_datas(ns, n_compare, show=True, save_dir=None):
    x = pd.date_range('2025-07-01', freq='MS', periods=24)
    ns = np.atleast_1d(ns)
    for n in ns:
        fig, (ax, ax2) = plt.subplots(2, figsize=(11,11))
        act_mean, outdata, difs = make_dataset(n, n_compare)

        for i in range(n_compare):
            ax.plot(x, outdata[i], c='r', alpha=0.5, ls='--')
        ax.plot(x, act_mean, c='b')
        ax.set_title('{} vs 10000'.format(n))
        ax.set_ylabel('PGR')
        ax.set_xlabel('PGR dif')
        ax2.boxplot(difs, labels=x)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(save_dir, '{}_comparison.png'.format(n)))
    if show:
        plt.show()


if __name__ == '__main__':
    plot_datas([1, 10, 100, 1000], 1000, save_dir=os.path.join(ksl_env.slmmac_dir_unbacked,
                                                               'test_full_model0000', 'comp_plots'))
