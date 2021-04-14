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
from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched, calc_impact_prob, \
    calc_cumulative_impact_prob


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


def plot_prob_impact(x, y, num, figsize):
    """

    :param x: probability data
    :param y: impact data
    :param num: number of bins for density estimate
    :param years: number of years to run this on
    :return:
    """
    fig, ax = plt.subplots(figsize=figsize)
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


def plot_all_1yr(site, mode, num=20, outdir=None, other_scen=None, other_scen_lbl='other storylines',
                 pt_labels=False):
    """
    plot all of the 1 year randoms +- other scenarios
    :param site: eyrewell or oxford
    :param mode: irrigated or dryland
    :param num: number of bins for the density plot
    :param outdir: None or directory (will be made) to save the plots to
    :param other_scen: none or dataframe with keys: 'plotlabel', 'pgr', 'pgra', 'prob'
    :param other_scen_lbl: a label for the other data for the legend, default is 'other storylines'
    :param pt_labels: bool if True label the other scens with the plotlabel scheme
    :return:
    """
    add_color = 'deeppink'
    plt_add = False
    if other_scen is not None:
        assert isinstance(other_scen, pd.DataFrame)
        assert np.in1d(['plotlabel', 'pgr', 'pgra', 'prob'], other_scen.columns).all()
        plt_add = True

    figsize = (10,8)
    plots = []
    ticks = [-16., -14., -12., -10., -8., -6., -4.]
    data = get_1yr_data(bad_irr=True, good_irr=True)
    data = data.dropna()
    # data = data.loc[np.isfinite(data.log10_prob)]  # todo why are some events prob infinite with new system, should have fixed
    prg, prob = get_pgr_prob_baseline_stiched(1, site, mode, irr_prop_from_zero=False)
    prg = prg / 1000
    x = data.log10_prob

    # pgra
    y = data[f'{site}-{mode}_pgra_yr1'] * -1 / 1000
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize)
    ax.set_title('Probability vs impact space for random 1 year stories\n')
    ax.set_ylabel('Pasture growth deficit tons DM/Ha/year from baseline\nimpact increases -->')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        ax.scatter(other_scen['prob'], other_scen['pgra'] / 1000 * -1, marker='^', c=add_color, label=other_scen_lbl)
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pgra']].itertuples(False, None):
                ax.annotate(l, (xi, yi / 1000 * -1))
    c = 'limegreen'
    ax.axhline(0, c=c, ls=':', label='baseline impact')
    ax.axvline(prob, c=c, ls=':', label='baseline probability')
    ax.legend()
    fig.tight_layout()
    plots.append(fig)

    # PGR
    y = data[f'{site}-{mode}_pg_yr1'] / 1000
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize)
    ax.set_title('Probability vs impact space for random 1 year stories\n')
    ax.set_ylabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        ax.scatter(other_scen['prob'], other_scen['pgr'] / 1000, marker='^', c=add_color, label=other_scen_lbl)
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pgr']].itertuples(False, None):
                ax.annotate(l, (xi, yi / 1000))
    c = 'limegreen'
    ax.axhline(prg, c=c, ls=':', label='baseline impact')
    ax.axvline(prob, c=c, ls=':', label='baseline probability')
    ax.legend()
    fig.tight_layout()
    plots.append(fig)

    # % PGR
    y = data[f'{site}-{mode}_pg_yr1'] / 1000 / prg * 100
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize)
    ax.set_title('Probability vs impact space for random 1 year stories\n')
    ax.set_ylabel('Percent Pasture growth tons DM/Ha/year\nimpact increases  <--')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        ax.scatter(other_scen['prob'], other_scen['pgr'] / 1000 / prg * 100, marker='^', c=add_color,
                   label=other_scen_lbl)
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pgr']].itertuples(False, None):
                ax.annotate(l, (xi, yi / prg / 1000 * 100))
    c = 'limegreen'
    ax.axhline(100, c=c, ls=':', label='baseline impact')
    ax.axvline(prob, c=c, ls=':', label='baseline probability')
    ax.legend()
    fig.tight_layout()
    plots.append(fig)

    # histogram prob + historgam for PGRA
    bins = 500
    y = data[f'{site}-{mode}_pg_yr1'] / 1000
    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize)
    ty, tx, t0 = ax1.hist(x, bins=bins)
    ax1.set_title('Probability Histogram')
    ax1.set_xlabel('Probability\nevents become more common -->')

    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'prob']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = ty.max() * (0.95 - 2 * i / 100)
                ax1.annotate(l, (xi, ylab_pos))

    ax1.axvline(prob, c=c, ls=':', label='baseline probability')
    ax1.set_ylabel('Count')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    ax1.legend()

    ty, tx, t0 = ax2.hist(y, bins=bins)
    ax2.set_xlabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    ax2.set_title('PG Histogram')
    ax2.set_ylabel('Count')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgr']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax2.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = ty.max() * (0.95 - 2 * i / 100)
                ax2.annotate(l, (xi / 1000, ylab_pos))

    ax2.axvline(prg, c=c, ls=':', label='baseline impact')
    ax2.legend()
    fig.tight_layout()
    plots.append(fig)

    step_size = 0.1
    cum_pgr, cum_prob = calc_impact_prob(pgr=np.concatenate((y, [prg])),
                                         prob=np.concatenate((x, [prob])), stepsize=step_size)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(cum_pgr, cum_prob, width=step_size / 2)
    ax1.set_title('Impact Probability Histogram')
    ax1.set_xlabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgr']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 2 * i / 100)
                ax1.annotate(l, (xi / 1000, ylab_pos))

    ax1.axvline(prg, c=c, ls=':', label='baseline pasture growth')
    ax1.set_ylabel('Probability of an event (sum to 1)')
    ax1.legend()
    fig.tight_layout()
    plots.append(fig)

    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=np.concatenate((y, [prg])),
                                                    prob=np.concatenate((x, [prob])), stepsize=step_size)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
    ax1.bar(cum_pgr, cum_prob, width=step_size / 2)
    ax1.set_title('Exceedance probability')
    ax1.set_xlabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgr']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 2 * i / 100)
                ax1.annotate(l, (xi / 1000, ylab_pos))

    ax1.axvline(prg, c=c, ls=':', label='baseline pasture growth')
    ax1.set_ylabel('Probability of an event with \nequal or greater Pasture growth')
    ax1.legend()

    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=np.concatenate((y, [prg])),
                                                    prob=np.concatenate((x, [prob])), stepsize=step_size,
                                                    more_production_than=False)
    ax2.bar(cum_pgr, cum_prob, width=step_size / 2)
    ax2.set_title('Non-exceedance probability')
    ax2.set_xlabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgr']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax2.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 2 * i / 100)
                ax2.annotate(l, (xi / 1000, ylab_pos))
    ax2.axvline(prg, c=c, ls=':', label='baseline pasture growth')
    ax2.set_ylabel('Probability of an event with \nequal or greater Pasture growth')
    ax2.legend()
    fig.tight_layout()
    plots.append(fig)

    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for fig in plots:
            plt.savefig(os.path.join(outdir, f'{fig.axes[0].title.get_text()}.png'))


# stats?, not now..., Though I could start digging into the reasons behind bad, and see what can reduce the impact!
# set of impacts? and then running that?


if __name__ == '__main__':
    data = get_1yr_data(bad_irr=True, good_irr=False)  # todo start by re-running this!!!
    data = data.dropna()
    data = data.loc[np.isfinite(data.log10_prob)]
    data.rename(columns={'ID': 'plotlabel', 'log10_prob': 'prob', 'eyrewell-irrigated_pg_yr1': 'pgr',
                         'eyrewell-irrigated_pgra_yr1': 'pgra'}, inplace=True)

    plot_all_1yr('eyrewell', 'irrigated', other_scen=data.iloc[0:5], other_scen_lbl='other storylines',
                 pt_labels=True)

    plt.show()
