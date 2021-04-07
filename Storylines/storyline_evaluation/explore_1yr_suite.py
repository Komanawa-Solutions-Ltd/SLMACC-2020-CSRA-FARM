"""
 Author: Matt Hanson
 Created: 6/04/2021 10:33 AM
 """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kde
import itertools
from  matplotlib.ticker import LogFormatterExponent


def make_density_xy(x, y, nbins=300):
    """
    made density raster
    :param x: x data
    :param y: y data
    :param nbins: number of bins along each axis so total number of bins == nbins**2
    :return: xi, yi, zi, args to put into pcolor mesh or contour
    """
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    return xi, yi, zi


if __name__ == '__main__':
    fig, ax = plt.subplots()
    num=20
    data = pd.read_hdf(r"D:\mh_unbacked\SLMACC_2020\pasture_growth_sims\random\IID_probs_pg_1y.hdf").dropna() # todo needs updating!
    x = data.log10_prob
    y = data['eyrewell-irrigated_yr1'] *-1
    ax.plot(x[0],y[0],c='k', label='Equal interval density')
    ax.scatter(x, y, alpha=0.2, label='Individual story')
    xi, yi, zi = make_density_xy(x, y, nbins=40)
    CS = ax.contour(xi, yi, np.log10(zi), levels=np.linspace(np.percentile(np.log10(zi), 50), np.max(zi),num),
                    cmap='magma')
    t = [10, 25, 50, 75, 90]
    xs, ys = [], []
    for p1, p2 in itertools.product(t, t):
        xs.append(np.percentile(x, p1))
        ys.append(np.percentile(y, p2))
    ax.scatter(xs, ys, c='r', label='1d quartiles: 10, 25, 50, 75, 90th')
    ax.legend()
    ax.set_title('Probability vs impact space for random 1 year stories\n')
    ax.set_ylabel('Pasture growth deficit kgDM/Ha/year from baseline\nimpact (deficit) increases -->')
    ax.set_xlabel('Probability\nevents become more common -->')
    ticks = [-16., -14., -12., -10.,  -8.,  -6.,  -4.]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{ {int(e)} }$" for e in ticks])

    plt.show()
