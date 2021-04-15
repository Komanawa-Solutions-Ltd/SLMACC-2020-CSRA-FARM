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
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite
from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched, calc_impact_prob, \
    calc_cumulative_impact_prob
import time


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


def plot_prob_impact(x, y, num, figsize, plt_data_density=True):
    """

    :param x: probability data
    :param y: impact data
    :param num: number of bins for density estimate
    :param figsize: figure size
    :param plt_data_density: boolean if True plot the data density contours, else do not (to save time)
    :return:
    """
    fig, ax = plt.subplots(figsize=figsize)
    if plt_data_density:
        ax.plot(x.mean(), y.mean(), c='k', label='Equal interval density')
    ax.scatter(x, y, alpha=0.2, label='Individual story')
    if plt_data_density:
        xi, yi, zi = make_density_xy(x, y, nbins=40)
        ax.contour(xi, yi, np.log10(zi), levels=np.linspace(np.percentile(np.log10(zi), 50), np.max(zi), num),
                   cmap='magma')
    t = [10, 25, 50, 75, 90]
    xs, ys = [], []
    for p1, p2 in itertools.product(t, t):
        xs.append(np.percentile(x, p1))
        ys.append(np.percentile(y, p2))
    ax.scatter(xs, ys, c='r', label='1d quartiles: 10, 25, 50, 75, 90th')

    return fig, ax


def plot_all_nyr(site, mode, nyr=1, num=20, outdir=None, other_scen=None, other_scen_lbl='other storylines',
                 pt_labels=False, plt_data_density=True, step_size=0.1):
    """
    plot all of the 1 year randoms +- other scenarios
    :param site: eyrewell or oxford
    :param mode: irrigated or dryland
    :param nyr: number of years in the sim... default 1 year
    :param num: number of bins for the density plot
    :param outdir: None or directory (will be made) to save the plots to
    :param other_scen: none or dataframe with keys: 'plotlabel', 'pgr', 'pgra', 'prob'
    :param other_scen_lbl: a label for the other data for the legend, default is 'other storylines'
    :param pt_labels: bool if True label the other scens with the plotlabel scheme
    :param plt_data_density: bool if true plot the data density contours, if False do not... saves time
    :param step_size: float the step size for binning the pasture growth data
    :return:
    """
    assert isinstance(nyr, int)
    add_color = 'deeppink'
    base_color = 'darkorange'
    base_ls = 'dashdot'
    base_lw = 2
    plt_add = False
    num_others = 0
    if other_scen is not None:
        assert isinstance(other_scen, pd.DataFrame)
        assert np.in1d(['plotlabel', 'pgr', 'pgra', 'prob'], other_scen.columns).all(), other_scen.columns
        plt_add = True
        num_others = len(other_scen)

    figsize = (16.5, 9.25)
    plots, plot_names = [], []
    if nyr == 1:
        data = get_1yr_data(bad_irr=True, good_irr=True)
    else:
        print('reading data')
        data = get_nyr_suite(nyr, site=site, mode=mode)  # .iloc[1:70000] # for testing

    data = data.dropna()
    prg, prob = get_pgr_prob_baseline_stiched(nyr, site, mode, irr_prop_from_zero=False)
    prg = prg / 1000
    x = data.loc[:, f'log10_prob_{mode}']
    if other_scen is None:
        xs = x
    else:
        t = other_scen['prob']
        xs = np.concatenate([x, t[np.isfinite(t)]])

    xmin = (np.floor(xs.min())) - 1
    xmax = (np.ceil(xs.max())) + 1
    space_between_tick = abs(round(((xmax - xmin) / 10) * 2) / 2)
    ticks = np.arange(xmin, xmax + space_between_tick, space_between_tick)

    # pgra
    _plot_pgra(data, site, mode, nyr, x, num, figsize, plt_data_density, ticks, plt_add, other_scen, add_color,
               other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob)

    # PGR
    _plt_pgr(data, site, mode, nyr, x, num, figsize, plt_data_density, ticks, plt_add, other_scen, add_color,
             other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob, prg)

    # % PGR
    _plt_pgr_per(data, site, mode, nyr, x, num, figsize, plt_data_density, ticks, plt_add, other_scen, add_color,
                 other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob, prg)

    # histogram prob + historgam for PGRA
    _plt_raw_hist(data, site, mode, nyr, x, figsize, num_others, ticks, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob)

    # plot impact probability
    _plot_impact_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                      other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, step_size)

    # plot cumulative probability
    _plt_cum_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, step_size)

    for fig in plots:
        fig.suptitle(f'{site}-{mode} {nyr} year long story lines'.capitalize())
        fig.tight_layout()

    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for fig, nm in zip(plots, plot_names):
            fig.savefig(os.path.join(outdir, f'{nm}.png'))


def _plot_pgra(data, site, mode, nyr, x, num, figsize, plt_data_density, ticks, plt_add, other_scen, add_color,
               other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob):
    print('plotting PGRA')
    start_time = time.time()
    y = data[f'{site}-{mode}_pgra_yr{nyr}'] * -1 / 1000
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize, plt_data_density=plt_data_density)
    ax.set_title(f'Probability vs impact for random stories')
    ax.set_ylabel('Pasture growth deficit tons DM/Ha/year from baseline\nimpact increases -->')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        ax.scatter(other_scen['prob'], other_scen['pgra'] / 1000 * -1, marker='^', c=add_color, label=other_scen_lbl)
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pgra']].itertuples(False, None):
                ax.annotate(l, (xi, yi / 1000 * -1))
    ax.axhline(0, c=base_color, lw=base_lw, ls=base_ls, label='baseline impact')
    ax.axvline(prob, c=base_color, lw=base_lw, ls=base_ls, label='baseline probability')
    ax.legend()

    plots.append(fig)
    plot_names.append(f'{site}-{mode}_pgra')
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_pgr(data, site, mode, nyr, x, num, figsize, plt_data_density, ticks, plt_add, other_scen, add_color,
             other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob, prg):
    print('plotting PGR')
    start_time = time.time()
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize, plt_data_density=plt_data_density)
    ax.set_title('Probability vs impact for random stories\n')
    ax.set_ylabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        ax.scatter(other_scen['prob'], other_scen['pgr'] / 1000, marker='^', c=add_color, label=other_scen_lbl)
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pgr']].itertuples(False, None):
                ax.annotate(l, (xi, yi / 1000))

    ax.axhline(prg, c=base_color, lw=base_lw, ls=base_ls, label='baseline impact')
    ax.axvline(prob, c=base_color, lw=base_lw, ls=base_ls, label='baseline probability')
    ax.legend()

    plots.append(fig)
    plot_names.append(f'{site}-{mode}_pgr')
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_pgr_per(data, site, mode, nyr, x, num, figsize, plt_data_density, ticks, plt_add, other_scen, add_color,
                 other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob, prg):
    print('plotting percent pgr')
    start_time = time.time()
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000 / prg * 100
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize, plt_data_density=plt_data_density)
    ax.set_title('Probability vs impact for random stories\n')
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
    ax.axhline(100, c=base_color, lw=base_lw, ls=base_ls, label='baseline impact')
    ax.axvline(prob, c=base_color, lw=base_lw, ls=base_ls, label='baseline probability')
    ax.legend()

    plots.append(fig)
    plot_names.append(f'{site}-{mode}_per_pgr')
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_raw_hist(data, site, mode, nyr, x, figsize, num_others, ticks, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, prob):
    print('plotting raw histogram')
    start_time = time.time()
    bins = 500
    y = data[f'{site}-{mode}_pgra_yr{nyr}'] / 1000
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
                ylab_pos = ty.max() * (0.95 - 90 / num_others * i / 100)
                ax1.annotate(l, (xi, ylab_pos))

    ax1.axvline(prob, c=base_color, lw=base_lw, ls=base_ls, label='baseline probability')
    ax1.set_ylabel('Count')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    ax1.legend()

    ty, tx, t0 = ax2.hist(y, bins=bins)
    ax2.set_xlabel('Pasture growth anomaly tons DM/Ha/year\nimpact increases <--')
    ax2.set_title('PG anomaly Histogram')
    ax2.set_ylabel('Count')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgra']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax2.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = ty.max() * (0.95 - 90 / num_others * i / 100)
                ax2.annotate(l, (xi / 1000, ylab_pos))

    ax2.axvline(0, c=base_color, lw=base_lw, ls=base_ls, label='baseline impact')
    ax2.legend()

    plots.append(fig)
    plot_names.append(f'{site}-{mode}_raw_hist')
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plot_impact_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                      other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, step_size):
    print('plotting impact probability')
    start_time = time.time()
    y = data[f'{site}-{mode}_pgra_yr{nyr}'] / 1000
    cum_pgr, cum_prob = calc_impact_prob(pgr=y,
                                         prob=x, stepsize=step_size)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(cum_pgr, cum_prob, width=step_size / 2)
    ax1.set_title('Impact Probability Histogram')
    ax1.set_xlabel('Pasture growth anomaly tons DM/Ha/year\nimpact increases <--')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgra']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 90 / num_others * i / 100)
                ax1.annotate(l, (xi / 1000, ylab_pos))

    ax1.axvline(0, c=base_color, lw=base_lw, ls=base_ls, label='baseline pasture growth')
    ax1.set_ylabel('Probability of an event (sum to 1)')
    ax1.legend()

    plots.append(fig)
    plot_names.append(f'{site}-{mode}_impact_prob')
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_cum_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, plots, plot_names, step_size):
    print('plot cumulative probability')
    start_time = time.time()
    y = data[f'{site}-{mode}_pgra_yr{nyr}'] / 1000
    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=y,
                                                    prob=x, stepsize=step_size)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
    ax1.bar(cum_pgr, cum_prob, width=step_size / 2)
    ax1.set_title('Exceedance probability')
    ax1.set_xlabel('Pasture growth anomaly tons DM/Ha/year\nimpact increases <--')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgra']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 90 / num_others * i / 100)
                ax1.annotate(l, (xi / 1000, ylab_pos))

    ax1.axvline(0, c=base_color, lw=base_lw, ls=base_ls, label='baseline pasture growth')
    ax1.set_ylabel('Probability of an event with \nequal or greater Pasture growth anomaly')
    ax1.legend()

    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=y,
                                                    prob=x, stepsize=step_size,
                                                    more_production_than=False)
    ax2.bar(cum_pgr, cum_prob, width=step_size / 2)
    ax2.set_title('Non-exceedance probability')
    ax2.set_xlabel('Pasture growth anomaly tons DM/Ha/year\nimpact increases <--')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pgra']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax2.axvline(xi / 1000, c=add_color, ls='--', label=tlab)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 90 / num_others * i / 100)
                ax2.annotate(l, (xi / 1000, ylab_pos))
    ax2.axvline(0, c=base_color, lw=base_lw, ls=base_ls, label='baseline pasture growth')
    ax2.set_ylabel('Probability of an event with \nequal or less Pasture growth anomaly')
    ax2.legend()

    plots.append(fig)
    plot_names.append(f'{site}-{mode}_exceed_prob')
    print(f'took {(time.time() - start_time) / 60} min to plot')


# stats?, not now..., Though I could start digging into the reasons behind bad, and see what can reduce the impact!
# set of impacts? and then running that?

if __name__ == '__main__':
    pass
