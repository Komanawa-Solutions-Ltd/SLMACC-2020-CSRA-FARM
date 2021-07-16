"""
 Author: Matt Hanson
 Created: 6/04/2021 10:33 AM
 """
import matplotlib.pyplot as plt
import os
import gc
import numpy as np
import pandas as pd
from scipy.stats import kde, binned_statistic_2d
import itertools
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite
from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched, calc_impact_prob, \
    calc_cumulative_impact_prob
import time
from copy import deepcopy


def make_density_xy(x, y, nbins=300):
    """
    made density raster
    :param x: x data
    :param y: y data
    :param nbins: number of bins along each axis so total number of bins == nbins**2
    :return: xi, yi, zi, args to put into pcolor mesh or contour
    """
    xi, yi = np.linspace(x.min(), x.max(), nbins), np.linspace(y.min(), y.max(), nbins)
    zi, f, f1, f2 = binned_statistic_2d(x, y, None, 'count', bins=[xi, yi])
    zi = zi.astype(float)
    zi[np.isclose(zi, 0)] = np.nan
    return xi, yi, (zi / np.nansum(zi)).transpose()


def plot_prob_impact(x, y, num, figsize):
    """

    :param x: probability data
    :param y: impact data
    :param num: number of levels for density estimate
    :param figsize: figure size
    :param plt_data_density: boolean if True plot the data density contours, else do not (to save time)
    :return:
    """
    fig, ax = plt.subplots(figsize=figsize)
    print('plotting scatter for prob v impact')
    print('finished plotting scatter for prob v impact')
    print('making data density')
    xi, yi, zi = make_density_xy(x, y, nbins=num)
    print('finished making density')
    edgecolors = 'face'
    linewidth = 0
    cm = ax.pcolormesh(xi, yi, zi * 100,
                       cmap='cividis', alpha=1)
    fig.colorbar(cm, extend='both', label='Data density (%)')
    t = [10, 25, 50, 75, 90]
    temp = np.array(list(itertools.product(t, t)))
    xs = np.percentile(x, temp[:, 0])
    ys = np.percentile(y, temp[:, 1])
    ax.scatter(xs, ys, c='r', label='1d quartiles: 10, 25, 50, 75, 90th')

    return fig, ax


def plot_impact_for_sites(data, num, figsize, correct=False):
    """

    :param x: probability data
    :param y: impact data
    :param num: number of levels for density estimate
    :param figsize: figure size
    :param plt_data_density: boolean if True plot the data density contours, else do not (to save time)
    :return:
    """
    assert isinstance(data, dict)

    keys = set(['*'.join(sorted([k1, k2])) for k1, k2 in itertools.product(data.keys(), data.keys())])
    keys = [e.split('*') for e in keys]
    out_figs, out_figids = [], []
    for k1, k2 in keys:
        if k1 == k2:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        print('making data density')
        x = data[k1]
        y = data[k2]
        xi, yi, zi = make_density_xy(x, y, nbins=num)
        print('finished making density')
        edgecolors = 'face'
        linewidth = 0
        cm = ax.pcolormesh(xi, yi, zi * 100,
                           cmap='magma', alpha=1)
        fig.colorbar(cm, extend='both', label='Data density (%)')
        ax.set_xlabel(f'pg: {k1} tons/yr')
        ax.set_ylabel(f'pg: {k2} tons/yr')
        out_figs.append(fig)
        out_figids.append(f'{k1}-{k2}')
    return out_figs, out_figids


def plot_all_nyr(site, mode, nyr=1, num=20, outdir=None, other_scen=None,
                 other_scen_lbl='other storylines', step_size=0.1,
                 pt_labels=False, close=False, additional_alpha=1, correct=False):
    """
    plot all of the 1 year randoms +- other scenarios
    :param site: eyrewell or oxford
    :param mode: irrigated or dryland
    :param nyr: number of years in the sim... default 1 year
    :param num: number of bins for the density plot
    :param outdir: None or directory (will be made) to save the plots to
    :param other_scen: none or dataframe with keys: 'plotlabel', 'pg', 'prob'
    :param other_scen_lbl: a label for the other data for the legend, default is 'other storylines'
    :param pt_labels: bool if True label the other scens with the plotlabel scheme
    :param step_size: float the step size for binning the pasture growth data
    :param close: bool if True close plot.
    :param correct: bool if True apply the DNZ correction
    :return:
    """
    assert isinstance(nyr, int)
    add_color = 'deeppink'
    base_color = 'limegreen'
    base_ls = 'dashdot'
    base_lw = 2
    plt_add = False
    num_others = 0
    if other_scen is not None:
        assert isinstance(other_scen, pd.DataFrame)
        assert np.in1d(['plotlabel', 'pg', 'prob'], other_scen.columns).all(), other_scen.columns
        plt_add = True
        num_others = len(other_scen)

    figsize = (16.5, 9.25)
    if nyr == 1:
        data = get_1yr_data(bad_irr=True, good_irr=True, correct=correct)
    else:
        print('reading data')
        data = get_nyr_suite(nyr, site=site, mode=mode, correct=correct)

    data = data.dropna()
    prg = get_pgr_prob_baseline_stiched(nyr, site, mode)
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
    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # PGR
    _plt_pgr(data, site, mode, nyr, x, num, figsize, ticks, plt_add, other_scen, add_color,
             other_scen_lbl, pt_labels, base_color, base_lw, base_ls, prg, outdir, close,
             additional_alpha=additional_alpha, correct=correct)

    # % PGR
    _plt_pgr_per(data, site, mode, nyr, x, num, figsize, ticks, plt_add, other_scen, add_color,
                 other_scen_lbl, pt_labels, base_color, base_lw, base_ls, prg, outdir, close,
                 additional_alpha=additional_alpha)

    # histogram prob + historgam for PGR
    _plt_raw_hist(data, site, mode, nyr, x, figsize, num_others, ticks, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, outdir, close,
                  additional_alpha=additional_alpha, correct=correct)

    # plot impact probability
    _plot_impact_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                      other_scen_lbl, pt_labels, base_color, base_lw, base_ls, step_size, outdir, close,
                      additional_alpha=additional_alpha, correct=correct)

    # plot cumulative probability
    _plt_cum_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, step_size, outdir, close,
                  additional_alpha=additional_alpha, correct=correct)

    # export key data
    if outdir is not None:
        y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
        cum_pgr, cum_prob = calc_impact_prob(pgr=y,
                                             prob=x, stepsize=step_size)
        temp = pd.DataFrame({'prob': cum_prob, 'pg': cum_pgr})
        temp.to_csv(os.path.join(outdir, f'{site}-{mode}_{nyr}yr_impact_prob.csv'))

        y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
        cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=y,
                                                        prob=x, stepsize=step_size)
        temp = pd.DataFrame({'prob': cum_prob, 'pg': cum_pgr})
        temp.to_csv(os.path.join(outdir, f'{site}-{mode}_{nyr}yr_cumulative_exceed_prob.csv'))


def _plot_pgra(data, site, mode, nyr, x, num, figsize, ticks, plt_add, other_scen, add_color,
               other_scen_lbl, pt_labels, base_color, base_lw, base_ls, outdir, close, additional_alpha):
    print('plotting PGRA')
    start_time = time.time()
    y = data[f'{site}-{mode}_pgra_yr{nyr}'].values * -1 / 1000
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize)
    ax.set_title(f'Probability vs impact for random stories')
    ax.set_ylabel('Pasture growth deficit tons DM/Ha/year from baseline\nimpact increases -->')
    ax.set_xlabel('Probability\nevents become more common -->')
    print('setting ticks')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        print('adding additional scen')
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pgra']].itertuples(False, None):
                tb = ax.annotate(l, (xi, yi / 1000 * -1))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))
        ax.scatter(other_scen['prob'], other_scen['pgra'] / 1000 * -1, marker='^',
                   s=60, c=add_color, alpha=additional_alpha, label=other_scen_lbl, zorder=10)

    ax.axhline(0, c=base_color, lw=base_lw, ls=base_ls, label='baseline impact')
    ax.legend()

    nm = (f'{site}-{mode}_pgra')
    fig.suptitle(f'{site}-{mode} {nyr} year long story lines'.capitalize())

    if outdir is not None:
        print(f'saving {nm}')
        fig.savefig(os.path.join(outdir, f'{nm}.png'))
        if close:
            plt.close(fig)
    else:
        if close:
            plt.show()
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_pgr(data, site, mode, nyr, x, num, figsize, ticks, plt_add, other_scen, add_color,
             other_scen_lbl, pt_labels, base_color, base_lw, base_ls, prg, outdir, close, additional_alpha, correct):
    print('plotting PGR')
    start_time = time.time()
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize)
    ax.set_title('Probability vs impact for random stories\n')
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
    ax.set_ylabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pg']].itertuples(False, None):
                tb = ax.annotate(l, (xi, yi / 1000))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))
        ax.scatter(other_scen['prob'], other_scen['pg'] / 1000, marker='^', c=add_color, alpha=additional_alpha,
                   label=other_scen_lbl,
                   zorder=10)

    if not correct:
        ax.axhline(prg, c=base_color, lw=base_lw, ls=base_ls, label='baseline impact')
        ax.legend()

    nm = f'{site}-{mode}_pg'

    fig.suptitle(f'{site}-{mode} {nyr} year long story lines'.capitalize())

    if outdir is not None:
        fig.savefig(os.path.join(outdir, f'{nm}.png'))
        if close:
            plt.close(fig)
    else:
        if close:
            plt.show()
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_pgr_per(data, site, mode, nyr, x, num, figsize, ticks, plt_add, other_scen, add_color,
                 other_scen_lbl, pt_labels, base_color, base_lw, base_ls, prg, outdir, close, additional_alpha,
                 ):
    print('plotting percent pgr')
    start_time = time.time()
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000 / prg * 100
    fig, ax = plot_prob_impact(x, y, num=num, figsize=figsize)
    ax.set_title('Probability vs impact for random stories\n')
    ax.set_ylabel('Percent Pasture growth tons DM/Ha/year\nimpact increases  <--')
    ax.set_xlabel('Probability\nevents become more common -->')
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    if plt_add:
        if pt_labels:
            for l, xi, yi in other_scen[['plotlabel', 'prob', 'pg']].itertuples(False, None):
                tb = ax.annotate(l, (xi, yi / prg / 1000 * 100))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))
        ax.scatter(other_scen['prob'], other_scen['pg'] / 1000 / prg * 100, marker='^', c=add_color,
                   alpha=additional_alpha,
                   label=other_scen_lbl, zorder=10)

    ax.axhline(100, c=base_color, lw=base_lw, ls=base_ls, label='baseline impact')
    ax.legend()

    nm = f'{site}-{mode}_per_pg'
    fig.suptitle(f'{site}-{mode} {nyr} year long story lines'.capitalize())
    if outdir is not None:
        fig.savefig(os.path.join(outdir, f'{nm}.png'))
        if close:
            plt.close(fig)
    else:
        if close:
            plt.show()
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_raw_hist(data, site, mode, nyr, x, figsize, num_others, ticks, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, outdir, close, additional_alpha,
                  correct):
    print('plotting raw histogram')
    start_time = time.time()
    bins = 500
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize)
    ty, tx, t0 = ax1.hist(x, bins=bins, color='grey')
    ax1.set_title('Probability Histogram')
    ax1.set_xlabel('Probability\nevents become more common -->')

    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'prob']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi, c=add_color, alpha=additional_alpha, ls='--', label=tlab, zorder=10)
            if pt_labels:
                ylab_pos = ty.max() * (0.95 - 90 / num_others * i / 100)
                tb = ax1.annotate(l, (xi, ylab_pos))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))

    ax1.set_ylabel('Count')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(["$10^{" + str(int(e)) + "}$" for e in ticks])
    ax1.legend()

    ty, tx, t0 = ax2.hist(y, bins=bins, color='grey')
    ax2.set_xlabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    ax2.set_title('PG anomaly Histogram')
    ax2.set_ylabel('Count')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pg']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax2.axvline(xi / 1000, c=add_color, alpha=additional_alpha, ls='--', label=tlab, zorder=10)
            if pt_labels:
                ylab_pos = ty.max() * (0.95 - 90 / num_others * i / 100)
                tb = ax2.annotate(l, (xi / 1000, ylab_pos))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))

    if not correct:
        ax2.axvline(get_pgr_prob_baseline_stiched(nyr, site, mode) / 1000, c=base_color, lw=base_lw, ls=base_ls,
                    label='baseline impact')
    ax2.legend()

    nm = f'{site}-{mode}_raw_hist'
    fig.suptitle(f'{site}-{mode} {nyr} year long story lines'.capitalize())
    fig.tight_layout()
    if outdir is not None:
        fig.savefig(os.path.join(outdir, f'{nm}.png'))
        if close:
            plt.close(fig)
    else:
        if close:
            plt.show()
    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plot_impact_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                      other_scen_lbl, pt_labels, base_color, base_lw, base_ls, step_size, outdir, close,
                      additional_alpha, correct):
    print('plotting impact probability')
    start_time = time.time()
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000

    cum_pgr, cum_prob = calc_impact_prob(pgr=y,
                                         prob=x, stepsize=step_size)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(cum_pgr, cum_prob, width=step_size / 2, color='grey')
    ax1.set_title('Impact Probability Histogram')
    ax1.set_xlabel('Pasture growth tons DM/Ha/year\nimpact increases <--')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pg']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi / 1000, c=add_color, alpha=additional_alpha, ls='--', label=tlab, zorder=10)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 90 / num_others * i / 100)
                tb = ax1.annotate(l, (xi / 1000, ylab_pos))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))

    if not correct:
        ax1.axvline(get_pgr_prob_baseline_stiched(nyr, site, mode) / 1000, c=base_color, lw=base_lw, ls=base_ls,
                    label='baseline pasture growth')
    ax1.set_ylabel('Probability of an event (sum to 1)')
    ax1.legend()

    nm = f'{site}-{mode}_impact_prob'
    fig.suptitle(f'{site}-{mode} {nyr} year long story lines'.capitalize())
    fig.tight_layout()
    if outdir is not None:
        fig.savefig(os.path.join(outdir, f'{nm}.png'))
        if close:
            plt.close(fig)
    else:
        if close:
            plt.show()

    print(f'took {(time.time() - start_time) / 60} min to plot')


def _plt_cum_prob(data, site, mode, nyr, x, figsize, num_others, plt_add, other_scen, add_color,
                  other_scen_lbl, pt_labels, base_color, base_lw, base_ls, step_size, outdir, close, additional_alpha,
                  correct):
    print('plot cumulative probability')
    start_time = time.time()
    y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=y,
                                                    prob=x, stepsize=step_size)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)
    ax1.bar(cum_pgr, cum_prob, width=step_size / 2, color='grey')
    ax1.set_title('Exceedance probability')

    ax1.set_xlabel('Pasture growth tons DM/Ha/year')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pg']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax1.axvline(xi / 1000, c=add_color, alpha=additional_alpha, ls='--', label=tlab, zorder=10)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 90 / num_others * i / 100)
                tb = ax1.annotate(l, (xi / 1000, ylab_pos))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))

    if not correct:
        ax1.axvline(get_pgr_prob_baseline_stiched(nyr, site, mode) / 1000, c=base_color, lw=base_lw, ls=base_ls,
                    label='baseline pasture growth')
    ax1.set_ylabel('Probability of an event with \nequal or greater Pasture growth')
    ax1.legend()

    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=y,
                                                    prob=x, stepsize=step_size,
                                                    more_production_than=False)
    ax2.bar(cum_pgr, cum_prob, width=step_size / 2, color='grey')
    ax2.set_title('Non-exceedance probability')
    ax2.set_xlabel('Pasture growth tons DM/Ha/year')
    if plt_add:
        for i, (l, xi) in enumerate(other_scen[['plotlabel', 'pg']].itertuples(False, None)):
            if i == 0:
                tlab = other_scen_lbl
            else:
                tlab = None
            ax2.axvline(xi / 1000, c=add_color, alpha=additional_alpha, ls='--', label=tlab, zorder=10)
            if pt_labels:
                ylab_pos = cum_prob.max() * (0.95 - 90 / num_others * i / 100)
                tb = ax2.annotate(l, (xi / 1000, ylab_pos))
                tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))

    if not correct:
        ax2.axvline(get_pgr_prob_baseline_stiched(nyr, site, mode) / 1000,
                    c=base_color, lw=base_lw, ls=base_ls, label='baseline pasture growth')
    ax2.set_ylabel('Probability of an event with \nequal or less Pasture growth')
    ax2.legend()

    nm = f'{site}-{mode}_exceed_prob'
    fig.suptitle(f'{site}-{mode} {nyr} year long story lines'.capitalize())
    fig.tight_layout()
    if outdir is not None:
        fig.savefig(os.path.join(outdir, f'{nm}.png'))
        if close:
            plt.close(fig)
    else:
        if close:
            plt.show()

    print(f'took {(time.time() - start_time) / 60} min to plot')


# stats?, not now..., Though I could start digging into the reasons behind bad, and see what can reduce the impact!
# set of impacts? and then running that?

if __name__ == '__main__':
    pass
