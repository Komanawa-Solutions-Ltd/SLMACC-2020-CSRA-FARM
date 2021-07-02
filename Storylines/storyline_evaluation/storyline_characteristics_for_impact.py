"""
 Author: Matt Hanson
 Created: 2/07/2021 9:51 AM
 """
import pickle

import numpy as np
import pandas as pd
from numbers import Number
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift

import ksl_env
from Storylines.storyline_runs.run_random_suite import get_1yr_data, default_mode_sites
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

name = 'random_'
base_story_dir = os.path.join(temp_storyline_dir, name)
cols = ['date', 'precip_class', 'temp_class', 'rest', 'rest_per', 'year',
        'month', 'precip_class_prev']


def get_suite(lower_bound, upper_bound, return_for_pca=False, state_limits=None):
    """
    get the storylines for the paths where data is between the two bounds
    :param lower_bound: dictionary {site-mode: None, float (for annual) or dictionary with at least 1 month limits}
    :param upper_bound: dictionary {site-mode: None, float (for annual) or dictionary with at least 1 month limits}
    :param return_for_pca: bool if True return as a 2d array for running through PCA, else return as a list for plotting
    :param state_limits: None or dictionary {month: ([precip_states], [temp_states], (rest_min, rest_max)), note
                         '*' can be passed for all possible for each.  only months with constraints need to be passed
    :return:
    """
    assert isinstance(state_limits, dict) or state_limits is None
    assert isinstance(lower_bound, dict) and isinstance(upper_bound, dict)
    assert set(lower_bound.keys()) == set(upper_bound.keys()) == {f'{s}-{m}' for m, s in default_mode_sites}
    impact_data = get_1yr_data(bad_irr=True, good_irr=True)
    impact_data = impact_data.dropna()
    impact_data.loc[:, 'sl_path'] = (base_story_dir + impact_data.loc[:, 'irr_type'] + '_irr/' +
                                     impact_data.loc[:, 'ID'] + '.csv')
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
            return pca_data[idx], data, impact_data.loc[idx]
        else:
            data = [pd.DataFrame(e, columns=cols) for e in data[idx]]
            return data, impact_data.loc[idx]
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
        return pca_data[idx][idx2], data, impact_data.loc[idx2]
    else:
        data = [pd.DataFrame(e, columns=cols) for e in data[idx2]]
        return data, impact_data.loc[idx2]


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


def run_plot_pca(data, impact_data, n_clusters=10, n_pcs=15, plot=True, show=False):
    print('running_pca')
    pca = PCA().fit(data)
    trans_data = pca.transform(data)
    print('explained variance')
    print(pca.explained_variance_ratio_)
    print('cumulative explained variance')
    print(np.cumsum(pca.explained_variance_ratio_))
    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(trans_data[:, 0:n_pcs])
    clusters = np.array(clusters)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(trans_data[:, 0], trans_data[:, 1], c=clusters, cmap='magma')
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')

        for mode, site in default_mode_sites:
            fig, ax = plt.subplots()
            ax.boxplot([impact_data.loc[clusters == k, f'{site}-{mode}_pg_yr1'] / 1000 for k in np.unique(clusters)])
            ax.set_title(f'{site}-{mode} - Agglomerative')
            ax.set_xlabel('cluster')
            ax.set_ylabel('pg growth tons DM/ha/yr')

        if show:
            plt.show()
    return clusters


if __name__ == '__main__':  # todo start working through this...
    # todo wrap the following into a single function and then talk to WS
    pca_data, data, impact_data = get_suite(lower_bound={
        'oxford-dryland': None,
        'eyrewell-irrigated': 10 * 1000,
        'oxford-irrigated': None,
    },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 14 * 1000,
            'oxford-irrigated': None,

        }, return_for_pca=True,
        state_limits={2: (['D'], '*', '*')}

    )
    print(len(data))
    clusters = run_plot_pca(pca_data, impact_data)

    from plot_storylines import plot_1_yr_storylines

    probs = 10 ** impact_data['log10_prob_irrigated']
    probs = probs / probs.sum()
    for c in np.unique(clusters):
        idx = clusters == c
        print(f'cluster: {c} prob: {probs[idx].sum()} size: {idx.sum()} norm_prob: {probs[idx].sum()/idx.sum():.3e}')
        plot_1_yr_storylines([e for e, i in zip(data, idx) if i], f'cluster {c}')
    plt.show()
