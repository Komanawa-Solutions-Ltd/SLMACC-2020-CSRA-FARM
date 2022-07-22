"""
created matt_dumont 
on: 21/07/22
"""
import numpy as np

from Storylines.storyline_runs.run_random_suite import get_1yr_data
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def make_clusters(site, mode):
    data = get_1yr_data().dropna()
    use_months = [12, 1, 2, 3, 4]
    use_data = data.loc[:, [f'{site}-{mode}_pg_m{m:02d}' for m in use_months]]
    pca = PCA(5)
    new_data = pca.fit_transform(use_data.values)
    print(pca.explained_variance_ratio_)
    for i in range(len(pca.explained_variance_ratio_)):
        print(i, np.cumsum(pca.explained_variance_ratio_[0:i + 1]).max())
    pass
    annual = data.loc[:, f'{site}-{mode}_pg_yr1'].values
    for c in range(5, 15):
        print(f'{c}_kmeans')
        k = KMeans(n_clusters=c)
        idxs = k.fit_predict(new_data[:, 0:3])
        fig, (ax, ax1) = plt.subplots(nrows=2)
        ax.scatter(new_data[:, 0], new_data[:, 1], c=idxs)
        ax.set_title(f'{c}_kmeans')

        bdata = [annual[idxs == i] for i in np.unique(idxs)]
        ax1.boxplot(bdata)

    plt.show()
    pass


def plot_clusters(nclusters, site, mode):
    data = get_1yr_data().dropna()
    key_months = [9, 10, 11, 12, 1, 2, 3, 4]
    use_months = [12, 1, 2, 3, 4]
    use_months = key_months
    use_data = data.loc[:, [f'{site}-{mode}_pg_m{m:02d}' for m in use_months]]
    pca = PCA(len(use_months))
    new_data = pca.fit_transform(use_data.values)
    print(pca.explained_variance_ratio_)
    for i in range(len(pca.explained_variance_ratio_)):
        print(i, np.cumsum(pca.explained_variance_ratio_[0:i + 1]).max())

    k = KMeans(n_clusters=nclusters)
    idxs = k.fit_predict(new_data[:, 0:3])
    fig, (ax, ax1) = plt.subplots(nrows=2)
    ax.scatter(new_data[:, 0], new_data[:, 1], c=idxs)
    ax.set_title(f'{nclusters}_kmeans')
    annual = data.loc[:, f'{site}-{mode}_pg_yr1'].values
    bdata = [annual[idxs == i] for i in np.unique(idxs)]
    ax1.boxplot(bdata)
    fig, axs = plt.subplots(3)
    fig, axs1 = plt.subplots(5)

    all_axs = np.concatenate((axs, axs1))
    for m, ax in zip(key_months, all_axs):
        bdata = [data[f'{site}-{mode}_pg_m{m:02d}'].values[idxs == i] for i in np.unique(idxs)]
        ax.boxplot(bdata)
        ax.set_title(f'month {m}')

    plt.show()


if __name__ == '__main__':
    #make_clusters('eyrewell', 'irrigated')
    plot_clusters(15, 'eyrewell', 'irrigated') # todo my favorite so far is 10 clusters
    # note including 9,10,11 months in the pca did not make much of a difference, variability in these months are not
    # very predicitve to the cluster.

    # todo worth looking at wether or not the storyline realisaions fall into different groups for the same storyline
    # todo could break the clusters into groups of good, moderate, bad
    # todo where does the vulnerabilities come from, typify the options
    # todo worth looking at the relative probability of each group!
    # todo at 15 groups are we seeing distinction between weather, and good/bad
    # todo comparison with unique events on a given month
    # todo good to look at the predicitve power of each month's impact on poduction values
    #  (e.g. I don't think that 9-11 is predicitive for eyrewell), variability? etc.
    #todo I need to look at the oxford data, does it show a different perspecitive
    # similar dryland???
