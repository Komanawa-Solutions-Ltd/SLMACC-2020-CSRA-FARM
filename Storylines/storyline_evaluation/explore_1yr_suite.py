"""
 Author: Matt Hanson
 Created: 6/04/2021 10:33 AM
 """
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.stats import kde
import itertools
from Storylines.storyline_runs.run_random_suite import get_1yr_data
from matplotlib.ticker import LogFormatterExponent
from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched


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

def plot_prob_impact(x,y,num, years=1):
    """

    :param x: probability data
    :param y: impact data
    :param num: number of bins for density estimate
    :param years: number of years to run this on
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(x[0], y[0], c='k', label='Equal interval density')
    ax.scatter(x, y, alpha=0.2, label='Individual story')
    xi, yi, zi = make_density_xy(x, y, nbins=40)
    CS = ax.contour(xi, yi, np.log10(zi), levels=np.linspace(np.percentile(np.log10(zi), 50), np.max(zi), num),
                    cmap='magma')
    t = [10, 25, 50, 75, 90]
    xs, ys = [], []
    for p1, p2 in itertools.product(t, t):
        xs.append(np.percentile(x, p1))
        ys.append(np.percentile(y, p2))
    ax.scatter(xs, ys, c='r', label='1d quartiles: 10, 25, 50, 75, 90th')


    return fig, ax

def plot_all_1yr(site, mode, num=20, outdir=None):
    # todo figsize
    plots = []
    ticks = [-16., -14., -12., -10., -8., -6., -4.]
    data = get_1yr_data(bad_irr=True, good_irr=False)
    data = data.dropna()
    data = data.loc[np.isfinite(data.log10_prob)] #todo why are some events prob infinite with new system
    prg, prob = get_pgr_prob_baseline_stiched(1, site, mode, irr_prop_from_zero=False)
    prg = prg/1000
    x = data.log10_prob

    # pgra
    y = data[f'{site}-{mode}_pgra_yr1'] * -1/ 1000
    fig,ax = plot_prob_impact(x,y,num)
    ax.set_title('Probability vs impact space for random 1 year stories\n')
    ax.set_ylabel('Pasture growth deficit tons DM/Ha/year from baseline\nimpact (deficit) increases -->')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    c = 'limegreen'
    ax.axhline(0, c=c, ls=':', label='baseline impact')
    ax.axvline(prob, c=c, ls=':', label='baseline probability')
    ax.legend()
    fig.tight_layout()
    plots.append(fig)

    # PGR
    y = data[f'{site}-{mode}_pg_yr1'] / 1000
    fig,ax = plot_prob_impact(x,y,num)
    ax.set_title('Probability vs impact space for random 1 year stories\n')
    ax.set_ylabel('Pasture growth tons DM/Ha/year\nimpact (deficit) increases <--')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    c = 'limegreen'
    ax.axhline(prg, c=c, ls=':', label='baseline impact')
    ax.axvline(prob, c=c, ls=':', label='baseline probability')
    ax.legend()
    fig.tight_layout()
    plots.append(fig)

    # % PGR
    y = data[f'{site}-{mode}_pg_yr1']/1000 / prg * 100
    fig,ax = plot_prob_impact(x,y,num)
    ax.set_title('Probability vs impact space for random 1 year stories\n')
    ax.set_ylabel('Percent Pasture growth tons DM/Ha/year\nimpact (deficit) increases  <--')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    c = 'limegreen'
    ax.axhline(100, c=c, ls=':', label='baseline impact')
    ax.axvline(prob, c=c, ls=':', label='baseline probability')
    ax.legend()
    fig.tight_layout()
    plots.append(fig)

    # histogram prob + historgam for PGRA
    bins = 500
    y = data[f'{site}-{mode}_pg_yr1']/1000
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.hist(x, bins=bins)
    ax1.set_title('Probability Histogram')
    ax1.set_xlabel('Probability\nevents become more common -->')
    ax1.axvline(prob, c=c, ls=':', label='baseline probability')
    ax1.set_ylabel('Count')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    ax1.legend()

    ax2.hist(y, bins=bins)
    ax2.set_xlabel('Pasture growth tons DM/Ha/year\nimpact (deficit) increases <--')
    ax2.set_title('PG Histogram')
    ax2.set_ylabel('Count')
    ax2.axvline(prg, c=c, ls=':', label='baseline impact')
    ax2.legend()
    fig.tight_layout()
    plots.append(fig)

    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for fig in plots:
            plt.savefig(os.path.join(outdir,f'{fig.axes[0].title.get_text()}.png'))



# todo stats?
# todo how to convert into 1 in 100 year styles... is this as simple as making a probability weighted (with repeats)
# set of impacts? and then running that?




if __name__ == '__main__':
    plot_all_1yr('eyrewell', 'irrigated')

    plt.show()
