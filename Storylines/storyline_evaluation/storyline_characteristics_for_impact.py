"""
 Author: Matt Hanson
 Created: 2/07/2021 9:51 AM
 """
import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
from numbers import Number
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from Storylines.storyline_evaluation.plot_storylines import plot_1_yr_storylines
from Storylines.storyline_evaluation.plot_nyr_suite import plot_impact_for_sites
import ksl_env
from Storylines.storyline_runs.run_random_suite import get_1yr_data, default_mode_sites
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Storylines.storyline_params import month_len

name = 'random_'
base_story_dir = os.path.join(temp_storyline_dir, name)
cols = ['date', 'precip_class', 'temp_class', 'rest', 'rest_per', 'year',
        'month', 'precip_class_prev']


def get_exceedence_prob():
    pass  # todo


def get_suite(lower_bound, upper_bound, return_for_pca=False, state_limits=None, correct=False):
    """
    get the storylines for the paths where data is between the two bounds
    :param lower_bound: dictionary {site-mode: None, float (for annual) or dictionary with at least 1 month limits}
    :param upper_bound: dictionary {site-mode: None, float (for annual) or dictionary with at least 1 month limits}
    :param return_for_pca: bool if True return as a 2d array for running through PCA, else return as a list for plotting
    :param state_limits: None or dictionary {month: ([precip_states], [temp_states], (rest_min, rest_max)), note
                         '*' can be passed for all possible for each.  only months with constraints need to be passed
    :param: correct: bool if True apply the DNZ correction
    :return:
    """
    assert isinstance(state_limits, dict) or state_limits is None
    assert isinstance(lower_bound, dict) and isinstance(upper_bound, dict)
    assert set(lower_bound.keys()) == set(upper_bound.keys()) == {f'{s}-{m}' for m, s in default_mode_sites}

    impact_data = get_1yr_data(bad_irr=True, good_irr=True, correct=correct)
    impact_data = impact_data.dropna()
    impact_data.loc[:, 'sl_path'] = (base_story_dir + impact_data.loc[:, 'irr_type'] + '_irr/' +
                                     impact_data.loc[:, 'ID'] + '.csv')

    full_prob = (10 ** impact_data.loc[:, ['log10_prob_irrigated', 'log10_prob_dryland']]).sum()

    idx = np.full((len(impact_data),), True)
    for mode, site in default_mode_sites:
        sm = f'{site}-{mode}'
        if lower_bound[sm] is None:
            assert upper_bound[sm] is None, f'both must be None for sm: {sm}'
            continue

        elif isinstance(lower_bound[sm], dict):
            assert isinstance(upper_bound[sm], dict)
            assert set(lower_bound[sm].keys()) == set(upper_bound[sm].keys())
            assert set(lower_bound[sm].keys()).issubset(set(range(1, 13)))
            for k in lower_bound[sm].keys():
                key = f'{sm}_pg_m{k:02d}'
                idx = idx & (impact_data[key] >= lower_bound[sm]) & (impact_data[key] <= upper_bound[sm])

        elif isinstance(lower_bound[sm], Number):
            assert isinstance(upper_bound[sm], Number)
            key = f'{sm}_pg_yr1'
            idx = idx & (impact_data[key] >= lower_bound[sm]) & (impact_data[key] <= upper_bound[sm])

        else:
            raise ValueError(f'unexpected type for lower_bound[{sm}]: {type(lower_bound[sm])}')

    # pull out data from the indexes and pre-prepared
    data = get_storyline_data(False)
    if state_limits is None:
        if return_for_pca:
            pca_data = get_storyline_data(True)
            data = [pd.DataFrame(e, columns=cols) for e in data[idx]]
            return pca_data[idx], data, impact_data.loc[idx], full_prob
        else:
            data = [pd.DataFrame(e, columns=cols) for e in data[idx]]
            return data, impact_data.loc[idx], full_prob
    data = data[idx]
    impact_data = impact_data[idx]
    idx2 = np.full((len(data),), True)
    assert set(state_limits.keys()).issubset(range(1, 13))
    for i, sl in enumerate(data):
        sl = pd.DataFrame(sl, columns=cols).set_index('month')
        temp = []
        for m, (pstate, tstate, restper) in state_limits.items():
            if pstate == '*':
                pstate = ['W', 'A', 'D']
            if tstate == '*':
                tstate = ['C', 'A', 'H']
            if restper == '*':
                restper = (0, 1)

            tidx = sl.loc[m, 'temp_class'] in tstate and sl.loc[m, 'precip_class'] in pstate
            tidx = tidx and restper[0] <= sl.loc[m, 'rest_per'] <= restper[1]
            temp.append(tidx)
        idx2[i] = all(temp)
    if return_for_pca:
        pca_data = get_storyline_data(True)
        data = [pd.DataFrame(e, columns=cols) for e in data[idx2]]
        return pca_data[idx][idx2], data, impact_data.loc[idx2], full_prob
    else:
        data = [pd.DataFrame(e, columns=cols) for e in data[idx2]]
        return data, impact_data.loc[idx2], full_prob


def get_storyline_data(pca=False):
    out_path_sl = os.path.join(ksl_env.mh_unbacked(r'Z2003_SLMACC\outputs_for_ws\norm'), 'random_storylines',
                               'random_pd.p')
    out_path_pca = os.path.join(ksl_env.mh_unbacked(r'Z2003_SLMACC\outputs_for_ws\norm'), 'random_storylines',
                                'random_pca.npy')
    if pca:
        return np.load(out_path_pca)
    else:
        return pickle.load(open(out_path_sl, 'rb'))


def make_all_storyline_data(calc_raw=False):
    out_path_sl = os.path.join(ksl_env.mh_unbacked(r'Z2003_SLMACC\outputs_for_ws\norm'), 'random_storylines',
                               'random_pd.p')
    out_path_pca = os.path.join(ksl_env.mh_unbacked(r'Z2003_SLMACC\outputs_for_ws\norm'), 'random_storylines',
                                'random_pca.npy')
    if not os.path.exists(os.path.dirname(out_path_pca)):
        os.makedirs(os.path.dirname(out_path_pca))

    impact_data = get_1yr_data(bad_irr=True, good_irr=True)
    impact_data = impact_data.dropna()
    impact_data.loc[:, 'sl_path'] = (base_story_dir + impact_data.loc[:, 'irr_type'] + '_irr/' +
                                     impact_data.loc[:, 'ID'] + '.csv')

    path_list = impact_data.loc[:, 'sl_path'].values
    if calc_raw:
        data = []
        for i, e in enumerate(path_list):
            if i % 1000 == 0:
                print(f'{i} of {len(path_list)}')

            data.append(pd.read_csv(e))
        data = np.array(data)
        pickle.dump(data, open(out_path_sl, 'wb'))

    print('putting data in pca format')
    replace_dict = {'precip_class': {'W': 0, 'A': 1, 'D': 2},
                    'temp_class': {'C': 0, 'A': 1, 'H': 2}}
    data = get_storyline_data()
    out = []

    for i, e in enumerate(data):
        if i % 1000 == 0:
            print(f'{i} of {len(path_list)}')
        t = pd.DataFrame(e, columns=cols)
        t = t[['precip_class', 'temp_class', 'rest']].replace(replace_dict)
        t = t.values.flatten()[np.newaxis]
        out.append(t)

    out = np.concatenate(out, axis=0).astype(float)
    np.save(out_path_pca, out, False, False)


def run_plot_pca(data, impact_data, n_clusters=20, n_pcs=15, plot=True, show=False, log_dir=None):
    log_text = []
    print('running_pca')
    pca = PCA().fit(data)
    trans_data = pca.transform(data)

    log_text.append('explained variance')
    log_text.append(' '.join([f'{e:.2e}' for e in pca.explained_variance_ratio_]))
    log_text.append('cumulative explained variance')
    log_text.append(' '.join([f'{e:.2e}' for e in np.cumsum(pca.explained_variance_ratio_)]))
    log_text.append(f'calculating {n_clusters} clusters for {n_pcs} principle components total '
                    f'explained variance{np.cumsum(pca.explained_variance_ratio_)[n_pcs - 1]}')

    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(trans_data[:, 0:n_pcs])
    clusters = np.array(clusters)

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'cluster_log.txt'), 'w') as f:
            f.write('\n'.join(log_text))

    if plot:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.scatter(trans_data[:, 0], trans_data[:, 1], c=clusters, cmap='magma')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')

        all_data = {}
        fig_all, axs = plt.subplots(3, sharex=True, figsize=(14, 7))

        for i, ((mode, site), ax_all) in enumerate(zip(default_mode_sites, axs)):
            fig, ax = plt.subplots(figsize=(14, 7))
            data = [impact_data.loc[:, f'{site}-{mode}_pg_yr1'] / 1000]
            all_data[f'{site}-{mode}'] = deepcopy(data[0])
            data = data + [impact_data.loc[clusters == k, f'{site}-{mode}_pg_yr1'] / 1000 for k in np.unique(clusters)]
            ax.boxplot(data,
                       labels=['all'] + [f'c:{i}' for i in range(n_clusters)])
            ax.set_title(f'{site}-{mode} - Agglomerative')
            ax.set_xlabel('cluster')
            ax.set_ylabel('pg growth tons DM/ha/yr')
            fig.savefig(os.path.join(log_dir, f'{site}-{mode}-pg.png'))
            ax_all.boxplot(data,
                           labels=['all'] + [f'c:{i}' for i in range(n_clusters)])
            ax_all.set_title(f'{site}-{mode} - Agglomerative')
            if i == 2:
                ax_all.set_xlabel('cluster')
            if i == 1:
                ax_all.set_ylabel('pg growth tons DM/ha/yr')
        fig_all.savefig(os.path.join(log_dir, f'all_sites-pg.png'))

        figs, figids = plot_impact_for_sites(all_data, 300, (14, 7))
        for fig, figid in zip(figs, figids):
            fig.suptitle('Full Suite')
            fig.tight_layout()
            fig.savefig(os.path.join(log_dir, f'site_comparison_all_{figid}.png'))

        plot_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        pg_fig, pg_axs = plt.subplots(nrows=3, figsize=(14, 10), sharex=True)
        for i, ((mode, site), ax) in enumerate(zip(default_mode_sites, pg_axs)):
            data = [impact_data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] / month_len[m] for m in plot_months]
            parts = ax.violinplot(data, positions=np.arange(1, len(plot_months) + 1),
                                  showmeans=False, showmedians=True, quantiles=[[0.25, 0.75] for e in plot_months])
            c = 'k'
            for pc in parts['bodies']:
                pc.set_facecolor(c)
            parts['cmedians'].set_color(c)
            parts['cquantiles'].set_color(c)
            parts['cmins'].set_color(c)
            parts['cmaxes'].set_color(c)
            parts['cbars'].set_color(c)

            ax.set_title(f'{site}-{mode}')
            if i == 2:
                ax.set_xlabel('Month')
                ax.set_xticks(np.arange(1, len(plot_months) + 1))
                ax.set_xticklabels([str(e) for e in plot_months])
                ax.set_xlim(0.5, len(plot_months) + 0.5)
            if i == 1:
                ax.set_ylabel('kg DM/ha/day')
        pg_fig.suptitle(f'Full Suite')
        pg_fig.tight_layout()
        pg_fig.savefig(os.path.join(log_dir, 'pg_curve_all.png'))

        for clust in np.unique(clusters):
            all_data = {}
            pg_fig, pg_axs = plt.subplots(nrows=3, figsize=(14, 10), sharex=True)
            for i, ((mode, site), ax) in enumerate(zip(default_mode_sites, pg_axs)):
                all_data[f'{site}-{mode}'] = impact_data.loc[clusters == clust, f'{site}-{mode}_pg_yr1'] / 1000
                data = [impact_data.loc[clusters == clust,
                                        f'{site}-{mode}_pg_m{m:02d}'] / month_len[m] for m in plot_months]
                parts = ax.violinplot(data, positions=np.arange(1, len(plot_months) + 1),
                                      showmeans=False, showmedians=True, quantiles=[[0.25, 0.75] for e in plot_months])
                c = 'k'
                for pc in parts['bodies']:
                    pc.set_facecolor(c)
                parts['cmedians'].set_color(c)
                parts['cquantiles'].set_color(c)
                parts['cmins'].set_color(c)
                parts['cmaxes'].set_color(c)
                parts['cbars'].set_color(c)

                ax.set_title(f'{site}-{mode}')
                if i == 2:
                    ax.set_xlabel('Month')
                    ax.set_xticks(np.arange(1, len(plot_months) + 1))
                    ax.set_xticklabels([str(e) for e in plot_months])
                    ax.set_xlim(0.5, len(plot_months) + 0.5)
                if i == 1:
                    ax.set_ylabel('kg DM/ha/day')
            pg_fig.suptitle(f'Cluster {clust:02d}')
            pg_fig.tight_layout()
            pg_fig.savefig(os.path.join(log_dir, f'pg_curve_clust_{clust:02d}.png'))

            figs, figids = plot_impact_for_sites(all_data, 300, (14, 7))
            for fig, figid in zip(figs, figids):
                fig.suptitle(f'Cluster {clust:02d}')
                fig.tight_layout()
                fig.savefig(os.path.join(log_dir, f'site_comparison_clust_{clust:02d}_{figid}.png'))

        if show:
            plt.show()
    return clusters


def run_plot_pca_correct(data, impact_data, n_clusters=20, n_pcs=15, plot=True, show=False, log_dir=None):
    log_text = []
    print('running_pca')
    pca = PCA().fit(data)
    trans_data = pca.transform(data)

    log_text.append('explained variance')
    log_text.append(' '.join([f'{e:.2e}' for e in pca.explained_variance_ratio_]))
    log_text.append('cumulative explained variance')
    log_text.append(' '.join([f'{e:.2e}' for e in np.cumsum(pca.explained_variance_ratio_)]))
    log_text.append(f'calculating {n_clusters} clusters for {n_pcs} principle components total '
                    f'explained variance{np.cumsum(pca.explained_variance_ratio_)[n_pcs - 1]}')

    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(trans_data[:, 0:n_pcs])
    clusters = np.array(clusters)

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'cluster_log.txt'), 'w') as f:
            f.write('\n'.join(log_text))

    if plot:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.scatter(trans_data[:, 0], trans_data[:, 1], c=clusters, cmap='magma')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')

        all_data = {}
        fig_all, axs = plt.subplots(3, sharex=True, figsize=(14, 7))

        for i, ((mode, site), ax_all) in enumerate(zip(default_mode_sites, axs)):
            fig, ax = plt.subplots(figsize=(14, 7))
            data = [impact_data.loc[:, f'{site}-{mode}_pg_yr1'] * 100]
            all_data[f'{site}-{mode}'] = deepcopy(data[0])
            data = data + [impact_data.loc[clusters == k, f'{site}-{mode}_pg_yr1'] * 100 for k in np.unique(clusters)]
            ax.boxplot(data,
                       labels=['all'] + [f'c:{i}' for i in range(n_clusters)])
            ax.set_title(f'{site}-{mode} - Agglomerative')
            ax.set_xlabel('cluster')
            ax.set_ylabel('pg growth percent of normal year')
            fig.savefig(os.path.join(log_dir, f'{site}-{mode}-pg.png'))
            ax_all.boxplot(data,
                           labels=['all'] + [f'c:{i}' for i in range(n_clusters)])
            ax_all.set_title(f'{site}-{mode} - Agglomerative')
            if i == 2:
                ax_all.set_xlabel('cluster')
            if i == 1:
                ax_all.set_ylabel('pg growth percent of normal year')

        fig_all.tight_layout()
        fig_all.savefig(os.path.join(log_dir, f'all_sites-pg.png'))

        figs, figids = plot_impact_for_sites(all_data, 300, (14, 7), True)
        for fig, figid in zip(figs, figids):
            fig.suptitle('Full Suite')
            fig.tight_layout()
            fig.savefig(os.path.join(log_dir, f'site_comparison_all_{figid}.png'))

        plot_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        pg_fig, pg_axs = plt.subplots(nrows=3, figsize=(14, 10), sharex=True)
        for i, ((mode, site), ax) in enumerate(zip(default_mode_sites, pg_axs)):
            data = [impact_data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] * 100 for m in plot_months]
            parts = ax.violinplot(data, positions=np.arange(1, len(plot_months) + 1),
                                  showmeans=False, showmedians=True, quantiles=[[0.25, 0.75] for e in plot_months])
            c = 'k'
            for pc in parts['bodies']:
                pc.set_facecolor(c)
            parts['cmedians'].set_color(c)
            parts['cquantiles'].set_color(c)
            parts['cmins'].set_color(c)
            parts['cmaxes'].set_color(c)
            parts['cbars'].set_color(c)

            ax.set_title(f'{site}-{mode}')
            if i == 0:
                ax.set_ylim(0, 400)
            if i == 2:
                ax.set_ylim(0, 200)
                ax.set_xlabel('Month')
                ax.set_xticks(np.arange(1, len(plot_months) + 1))
                ax.set_xticklabels([str(e) for e in plot_months])
                ax.set_xlim(0.5, len(plot_months) + 0.5)
            if i == 1:
                ax.set_ylim(0, 200)
                ax.set_ylabel('percent of normal year monthly growth')
        pg_fig.suptitle(f'Full Suite')
        pg_fig.tight_layout()
        pg_fig.savefig(os.path.join(log_dir, 'pg_curve_all.png'))

        for clust in np.unique(clusters):
            all_data = {}
            pg_fig, pg_axs = plt.subplots(nrows=3, figsize=(14, 10), sharex=True)
            for i, ((mode, site), ax) in enumerate(zip(default_mode_sites, pg_axs)):
                all_data[f'{site}-{mode}'] = impact_data.loc[clusters == clust, f'{site}-{mode}_pg_yr1'] * 100
                data = [impact_data.loc[clusters == clust,
                                        f'{site}-{mode}_pg_m{m:02d}'] * 100 for m in plot_months]
                parts = ax.violinplot(data, positions=np.arange(1, len(plot_months) + 1),
                                      showmeans=False, showmedians=True, quantiles=[[0.25, 0.75] for e in plot_months])
                c = 'k'
                for pc in parts['bodies']:
                    pc.set_facecolor(c)
                parts['cmedians'].set_color(c)
                parts['cquantiles'].set_color(c)
                parts['cmins'].set_color(c)
                parts['cmaxes'].set_color(c)
                parts['cbars'].set_color(c)

                ax.set_title(f'{site}-{mode}')
                if i == 0:
                    ax.set_ylim(0, 400)  # todo check!
                if i == 2:
                    ax.set_ylim(0, 200)
                    ax.set_xlabel('Month')
                    ax.set_xticks(np.arange(1, len(plot_months) + 1))
                    ax.set_xticklabels([str(e) for e in plot_months])
                    ax.set_xlim(0.5, len(plot_months) + 0.5)
                if i == 1:
                    ax.set_ylim(0, 200)
                    ax.set_ylabel('percent of normal years monthly growth')
            pg_fig.suptitle(f'Cluster {clust:02d}')
            pg_fig.tight_layout()
            pg_fig.savefig(os.path.join(log_dir, f'pg_curve_clust_{clust:02d}.png'))

            figs, figids = plot_impact_for_sites(all_data, 300, (14, 7), True)
            for fig, figid in zip(figs, figids):
                fig.suptitle(f'Cluster {clust:02d}')
                fig.tight_layout()
                fig.savefig(os.path.join(log_dir, f'site_comparison_clust_{clust:02d}_{figid}.png'))

        if show:
            plt.show()
    return clusters


def storyline_subclusters(outdir, lower_bound, upper_bound, state_limits=None, n_clusters=20, n_pcs=15,
                          save_stories=True, correct=False):
    """

    :param outdir:
    :param lower_bound:
    :param upper_bound:
    :param state_limits:
    :param n_clusters:
    :param n_pcs:
    :param save_stories:
    :param correct: bool, if True then apply DNZ correction
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pca_data, data, impact_data, full_prob = get_suite(lower_bound=lower_bound,
                                                       upper_bound=upper_bound,
                                                       return_for_pca=True,
                                                       state_limits=state_limits,
                                                       correct=correct
                                                       )
    print(len(data))
    total_prob = (10 ** impact_data.loc[:, ['log10_prob_irrigated', 'log10_prob_dryland']]).sum() / full_prob
    exceedence = {}
    for mode, site in default_mode_sites:
        if correct:
            exceedence[f'{site}-{mode}'] = pd.read_csv(
                os.path.join(ksl_env.slmmac_dir, r"outputs_for_ws\norm",
                             r"random_scen_plots\1yr_correct", f"{site}-{mode}_1yr_cumulative_exceed_prob.csv"))

            impact = impact_data.loc[:, f'{site}-{mode}_pg_yr1'].median() * 100
            probs = exceedence[f'{site}-{mode}'].prob.values
            impacts = exceedence[f'{site}-{mode}'].pg.values
        else:
            exceedence[f'{site}-{mode}'] = pd.read_csv(
                os.path.join(ksl_env.slmmac_dir, r"outputs_for_ws\norm",
                             r"random_scen_plots\1yr", f"{site}-{mode}_1yr_cumulative_exceed_prob.csv"))

            impact = impact_data.loc[:, f'{site}-{mode}_pg_yr1'].median() / 1000
            probs = exceedence[f'{site}-{mode}'].prob.values
            impacts = exceedence[f'{site}-{mode}'].pg.values
        idx = np.argmin(np.abs(impacts - impact))
        total_prob.loc[f'higher_pg_prob_{site}-{mode}_median'] = probs[idx]
        total_prob.loc[f'lower_pg_prob_{site}-{mode}_median'] = 1 - probs[idx]

    total_prob.to_csv(os.path.join(outdir, 'explained_probability.csv'))
    if correct:
        clusters = run_plot_pca_correct(pca_data, impact_data, n_clusters=n_clusters, n_pcs=n_pcs, log_dir=outdir)
    else:
        clusters = run_plot_pca(pca_data, impact_data, n_clusters=n_clusters, n_pcs=n_pcs, log_dir=outdir)
    impact_data.loc[:, 'cluster'] = clusters
    impact_data.to_csv(os.path.join(outdir, 'prop_pg_cluster_data.csv'))

    probs = 10 ** impact_data['log10_prob_irrigated']
    probs = probs / probs.sum()
    cluster_data = pd.DataFrame(index=np.unique(clusters), columns=['rel_cum_prob', 'size', 'norm_prob'])
    cluster_data.index.name = 'cluster'
    temp_data, precip_data, rest_data = plot_1_yr_storylines(data, f'all storylines',
                                                             outdir=outdir)

    for c in np.unique(clusters):
        idx = clusters == c
        cluster_data.loc[c, 'rel_cum_prob'] = probs[idx].sum()
        cluster_data.loc[c, 'size'] = idx.sum()
        cluster_data.loc[c, 'norm_prob'] = probs[idx].sum() / idx.sum()

        for mode, site in default_mode_sites:
            if correct:
                impact_v = impact_data.loc[impact_data.loc[:, 'cluster'] == c, f'{site}-{mode}_pg_yr1'].median() * 100
            else:
                impact_v = impact_data.loc[impact_data.loc[:, 'cluster'] == c, f'{site}-{mode}_pg_yr1'].median() / 1000
            probs_value = exceedence[f'{site}-{mode}'].prob.values
            impacts_value = exceedence[f'{site}-{mode}'].pg.values
            idx_value = np.argmin(np.abs(impacts_value - impact_v))
            cluster_data.loc[c, f'higher_pg_prob_{site}-{mode}_median'] = probs_value[idx_value]
            cluster_data.loc[c, f'lower_pg_prob_{site}-{mode}_median'] = 1 - probs_value[idx_value]

        use_data = [e for e, i in zip(data, idx) if i]
        temp_data, precip_data, rest_data = plot_1_yr_storylines(use_data, f'cluster {c}',
                                                                 outdir=outdir)
        if save_stories:
            sl_dir = os.path.join(outdir, f'storylines_cluster_{c:03d}')
            if not os.path.exists(sl_dir):
                os.makedirs(sl_dir)
            for nm, sl in zip(impact_data.loc[idx, 'ID'], use_data):
                sl.to_csv(os.path.join(sl_dir, f'{nm}.csv'))
    cluster_data.loc[:, 'full_prob_irr'] = cluster_data.loc[:, 'rel_cum_prob'] * total_prob.loc['log10_prob_irrigated']
    cluster_data.loc[:, 'full_prob_dry'] = cluster_data.loc[:, 'rel_cum_prob'] * total_prob.loc['log10_prob_dryland']
    cluster_data.to_csv(os.path.join(outdir, 'cluster_data.csv'))
    plt.close('all')


if __name__ == '__main__':
    storyline_subclusters(
        outdir=r"C:\Users\dumon\Downloads\test_storyline_subcluster",
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 10 * 1000,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 14 * 1000,
            'oxford-irrigated': None,

        },

        state_limits={2: (['D'], '*', '*')},
        n_clusters=20,
        n_pcs=15,
        save_stories=True
    )
