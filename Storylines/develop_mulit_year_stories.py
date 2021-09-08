"""
 Author: Matt Hanson
 Created: 6/09/2021 10:16 AM
 """
import itertools
import shutil
import glob
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from Storylines.storyline_params import month_len
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from Storylines.storyline_runs.run_random_suite import get_1yr_data
from Storylines.storyline_evaluation.storyline_characteristics_for_impact import add_exceedence_prob
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp, pgm_log_dir, \
    default_mode_sites


def make_multi_year_stories_from_random_suite(outdir, year_stories, n, start_seed=1156854, re_save_org_stories=False):
    """

    :param outdir: directory to save the storylines and prob files
    :param year_stories: dictionary with keys int (e.g. year 0,1,2....): vals list of paths to storylines
    :param n: number of storylines to create
    :param start_seed: the seed that starts it all, default is set
    :return:
    """
    unlinked_outdir, linked_outdir = os.path.join(outdir, 'unlinked'), os.path.join(outdir, 'linked')
    for d in unlinked_outdir, linked_outdir:
        if not os.path.exists(d):
            os.makedirs(d)

    assert isinstance(year_stories, dict)
    assert all([isinstance(k, int) for k in year_stories.keys()])
    assert set(year_stories.keys()) == set(range(max(year_stories.keys()) + 1))

    np.random.seed(start_seed)
    all_seeds = np.random.randint(1151, 1687532, 5 * len(year_stories.keys()))

    # set up all storyline data
    storyline_data = get_1yr_data(bad_irr=True, good_irr=True, correct=True)
    storyline_data.loc[:, 'full_ID'] = storyline_data.loc[:, 'ID'] + '_' + storyline_data.loc[:, 'irr_type']
    storyline_data.set_index('full_ID', inplace=True)

    all_stories = []
    for v in year_stories.values():
        all_stories.extend(v)
    all_stories = np.array(all_stories)
    stories_exist = np.array([os.path.exists(s) for s in all_stories])

    # save orginal storylines

    temp = np.unique(all_stories)
    for i, p in enumerate(temp):
        if i % 100 == 0:
            print(f'saving original storyline {i} of {len(temp)}')
        dist = os.path.join(unlinked_outdir, os.path.basename(p))

        if re_save_org_stories:
            if os.path.exists(dist):
                os.remove(dist)

        if not os.path.exists(p):
            shutil.copy(p, dist)

    assert all(stories_exist), 'stories do not exist:\n' + '\n'.join(all_stories[~stories_exist])
    storyline_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_stories]
    idx = np.in1d(storyline_ids, storyline_data.index)
    assert idx.all(), 'missing storyline ids:\n' + '\n'.join(storyline_ids[~idx])

    outdata = pd.DataFrame(index=range(n))
    outdata.loc[:, 'ID'] = [f'mrsl-{i:06d}' for i in outdata.index]
    output_storylines = [[] for i in range(n)]
    # set up random suite
    for i, (y, stories) in enumerate(year_stories.items()):
        # random selection
        np.random.seed(all_seeds[i])
        temp_storylines = np.random.choice(stories, n)
        temp_storyline_ids = np.array([os.path.splitext(os.path.basename(p))[0] for p in temp_storylines])

        # capture key data
        outdata.loc[:, f'rid_yr_{y:02d}'] = temp_storyline_ids
        outdata.loc[:, f'log10_true_prob_irr-{y:02d}'] = storyline_data.loc[temp_storyline_ids,
                                                                            'log10_prob_irrigated'].values
        outdata.loc[:, f'log10_true_prob_dry-{y:02d}'] = storyline_data.loc[temp_storyline_ids,
                                                                            'log10_prob_dryland'].values
        for j, slp in enumerate(temp_storylines):
            t = pd.read_csv(slp)
            t.loc[:, 'year'] += i
            t.loc[:, 'date'] = pd.to_datetime(t.loc[:, 'date']) + pd.offsets.DateOffset(years=i)
            output_storylines[j].append(t)

    # save storylines
    print('saving storylines')
    for i, sl in enumerate(output_storylines):
        pd.concat(sl).to_csv(os.path.join(linked_outdir, f'mrsl-{i:06d}.csv'))

    # calc total probs
    temp = outdata.loc[:, [f'log10_true_prob_irr-{y:02d}' for y in year_stories.keys()]].sum(axis=1)
    outdata.loc[:, f'log10_true_prob_irr-all'] = temp
    temp = outdata.loc[:, [f'log10_true_prob_dry-{y:02d}' for y in year_stories.keys()]].sum(axis=1)
    outdata.loc[:, f'log10_true_prob_dry-all'] = temp

    outdata.set_index('ID', inplace=True)
    outdata.to_hdf(os.path.join(outdir, 'id_probs.hdf'), 'prob', mode='w')
    outdata.to_csv(os.path.join(outdir, 'id_probs.csv'))

    # save a params textfile...make sure it does not conflict with reading the storylines
    with open(os.path.join(outdir, 'params.txt'), 'w') as f:
        f.write(f'nyears = {len(year_stories.keys())}\n')
        f.write(f'years = {year_stories.keys()}\n')
        f.write(f'start_seed = {start_seed}\n')
        f.write(f'number of sims (n) = {n}\n')


def run_multi_year_pg_model(storyline_dir, data_dir, name, desc, seed=1582354):  # todo check!
    unlinked_storyline_dir = os.path.join(storyline_dir, 'unlinked')
    linked_storyline_dir = os.path.join(storyline_dir, 'linked')
    unlinked_data_dir, linked_data_dir = os.path.join(data_dir, 'unlinked'), os.path.join(data_dir, 'linked')

    print('running linked')
    run_stories = glob.glob(os.path.join(linked_storyline_dir, 'mrsl-*.csv'))
    outdirs = [linked_data_dir for e in run_stories]

    run_full_model_mp(
        storyline_path_mult=run_stories,
        outdir_mult=outdirs,
        nsims_mult=96,
        log_path=os.path.join(pgm_log_dir, f'{name}-linked'),
        description_mult=desc,
        padock_rest_mult=False,
        save_daily_mult=False,
        verbose=False,
        mode_sites_mult=default_mode_sites,
        re_run=False,  # and additional safety
        seed=seed,
        use_1_seed=True,
    )

    print('running non-linked')
    run_stories = glob.glob(os.path.join(unlinked_storyline_dir, 'rsl-*.csv'))
    outdirs = [unlinked_data_dir for e in run_stories]

    run_full_model_mp(
        storyline_path_mult=run_stories,
        outdir_mult=outdirs,
        nsims_mult=96,
        log_path=os.path.join(pgm_log_dir, f'{name}-unlinked'),
        description_mult=desc,
        padock_rest_mult=False,
        save_daily_mult=False,
        verbose=False,
        mode_sites_mult=default_mode_sites,
        re_run=False,  # and additional safety
        seed=seed,
        use_1_seed=True,
    )


def create_pg_data_multi_year(storyline_dir, data_dir, outpath):
    """

    :param storyline_dir: directory with the storylines
    :param data_dir_linked: directory with run basgra models
    :param outpath: outpath to save the data (without extension) both csv and hdf will be saved.
    :return:
    """
    unlinked_data_dir, linked_data_dir = os.path.join(data_dir, 'unlinked'), os.path.join(data_dir, 'linked')
    print('creating 1yr pg data')
    data = pd.read_hdf(os.path.join(storyline_dir, 'id_probs.hdf'), 'prob')
    with open(os.path.join(storyline_dir, 'params.txt')) as f:
        n_years = int(f.readline().split('=')[-1])
    assert isinstance(data, pd.DataFrame)

    mkeys = []
    m_lens = []
    for y in range(n_years):
        mkeys.extend([f'yr{y:02d}-m{m:02d}' for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]])
        m_lens.extend([31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30])
    m_lens = np.array(m_lens)
    for site, mode in default_mode_sites:
        key = f'{mode}-{site}'
        data.loc[:, f'{key}-linked_pg_all'] = np.nan
        for y in range(n_years):
            data.loc[:, f'{key}-linked_pg_yr{y:02d}'] = np.nan

        for mk in mkeys:
            data.loc[:, f'{key}-linked_pg_{mk}'] = np.nan

        for i, idv in enumerate(data.index):
            if i % 1000 == 0:
                print(f'starting to read sim {i} for site: {site} and mode: {mode}')

            p = os.path.join(linked_data_dir, f'{idv}-{key}.nc')
            if not os.path.exists(p):
                continue

            nc_data = nc.Dataset(p)
            temp = np.array(nc_data.variables['m_PGR'])
            temp *= m_lens[:, np.newaxis]
            temp = np.nanmean(temp, axis=1)

            for j, mk in enumerate(mkeys):
                data.loc[idv, f'{key}-linked_pg_{mk}'] = temp[j]
            nc_data.close()

        for y in range(n_years):
            temp = data.loc[:, [f'{key}-linked_pg_yr{y:02d}-m{m:02d}' for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]]]
            data.loc[:, f'{key}-linked_pg_yr{y:02d}'] = temp.sum(axis=1)
        temp = data.loc[:, [f'{key}-linked_pg_yr{y:02d}' for y in range(n_years)]]
        data.loc[:, f'{key}-linked_pg_all'] = temp.sum(axis=1)

    data = extract_non_linked_data(data)

    data.to_hdf(os.path.join(storyline_dir, f'id_probs.hdf'), 'prob', mode='w')
    data.to_csv(os.path.join(storyline_dir, f'id_probs.csv'))
    data.to_hdf(f'{outpath}.hdf', 'prob', mode='w')
    data.to_hdf(f'{outpath}.csv', 'prob')


def extract_non_linked_data(data, years, unlinked_pg_dir):  # todo test this
    # extract raw data
    uninked_story_data = pd.DataFrame({
        'ID': np.unique(data.loc[:, [f'rid_yr_{y:02d}' for y in range(years)]].values.flatten())
    })

    storyline_data = _read_non_linked(uninked_story_data, unlinked_pg_dir)
    storyline_data = add_exceedence_prob(storyline_data, correct=True, impact_in_tons=False)
    for mode, site in default_mode_sites:
        storyline_data.loc[:, f'log10_non_exceed_prob_{site}-{mode}'] = np.log10(
            storyline_data.loc[:, f'non-exceed_prob_per_{site}-{mode}'] / 100)
    storyline_data.set_index('ID', inplace=True)

    # pull out the data needed
    for y in range(years):
        temp_storyline_ids = data.loc[:, f'rid_yr_{y:02d}']
        for mode, site in default_mode_sites:
            data.loc[:, f'log10_non_exceed_prob_{site}-{mode}-{y:02d}'] = storyline_data.loc[
                temp_storyline_ids, f'log10_non_exceed_prob_{site}-{mode}'].values
            data.loc[:, f'{site}-{mode}-not_linked_pg_yr{y:02d}'] = storyline_data.loc[
                temp_storyline_ids, f'{site}-{mode}_pg_yr1'].values

            # pull in non-linked data (for direct comparison)
            for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]:
                data.loc[:, f'{site}-{mode}-not_linked_pg_yr{y:02d}-m{m:02d}'] = storyline_data.loc[
                    temp_storyline_ids, f'{site}-{mode}_pg_m{m:02d}'].values

    for mode, site in default_mode_sites:
        temp = data.loc[:, [f'log10_non_exceed_prob_{site}-{mode}-{y:02d}' for y in range(years)]]
        data.loc[:, f'log10_non_exceed_prob_{site}-{mode}-all'] = temp.sum(axis=1)
        temp = data.loc[:, [f'{site}-{mode}-not_linked_pg_yr{y:02d}' for y in range(years)]]
        data.loc[:, f'{site}-{mode}-not_linked_pg-all'] = temp.sum(axis=1)

    return data


def _read_non_linked(data, pg_dir):
    """

    :param bad_irr: bool if True then create irrigation from 50-99th percentile if False 1-50th percentile
    :return:
    """
    assert isinstance(data, pd.DataFrame)
    for site, mode in default_mode_sites:
        key = f'{mode}-{site}'
        data.loc[:, f'{key}_pg_yr1'] = np.nan
        for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]:
            data.loc[:, f'{key}_pg_m{m:02d}'] = np.nan

        for i, idv in data.loc[:, ['ID']].itertuples(True, None):
            if i % 1000 == 0:
                print(f'starting to read sim {i} for site: {site} and mode: {mode}')
            p = os.path.join(pg_dir, f'{idv}-{key}.nc')
            if not os.path.exists(p):
                continue

            nc_data = nc.Dataset(p)
            temp = np.array(nc_data.variables['m_PGR'])
            temp *= np.array([31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30])[:, np.newaxis]
            temp = np.nanmean(temp, axis=1)
            for j, m in enumerate([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]):
                data.loc[i, f'{key}_pg_m{m:02d}'] = temp[j]
            temp = temp.sum()
            data.loc[i, f'{key}_pg_yr1'] = temp  #
            nc_data.close()
    return data


def plot_muli_year_total(site_modes, impact_data):  # todo!
    raise NotImplementedError


def plot_multi_year_monthly(outpath, mode_sites, impact_data, nyears, sup_title, show=False):  # todo check
    plot_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    mks = []
    c_non_linked = 'indianred'
    c_linked = 'royalblue'

    for y in range(nyears):
        mks.extend([f'yr{y:02d}-m{m:02d}' for m in plot_months])

    pg_fig, pg_axs = plt.subplots(nrows=len(mode_sites), figsize=(14, 10), sharex=True)
    for i, ((mode, site), ax) in enumerate(zip(mode_sites, pg_axs)):
        # plot non-linked data
        data = [impact_data.loc[
                :,
                f'{site}-{mode}-not_linked_pg_{mk}'] / month_len[int(mk[-2:])] for mk in mks
                ]
        parts = ax.violinplot(data, positions=np.arange(1, len(plot_months) * nyears + 1),
                              showmeans=False, showmedians=True,
                              quantiles=[[0.25, 0.75] for e in plot_months * nyears])
        for pc in parts['bodies']:
            pc.set_facecolor(c_non_linked)
        parts['cmedians'].set_color(c_non_linked)
        parts['cquantiles'].set_color(c_non_linked)
        parts['cmins'].set_color(c_non_linked)
        parts['cmaxes'].set_color(c_non_linked)
        parts['cbars'].set_color(c_non_linked)

        # plot linked data
        data = [impact_data.loc[
                :,
                f'{site}-{mode}-linked_pg_{mk}'] / month_len[int(mk[-2:])] for mk in mks
                ]
        parts = ax.violinplot(data, positions=np.arange(1, len(plot_months) * nyears + 1) + 0.5,  # todo offsets?
                              showmeans=False, showmedians=True,
                              quantiles=[[0.25, 0.75] for e in plot_months * nyears])
        for pc in parts['bodies']:
            pc.set_facecolor(c_linked)
        parts['cmedians'].set_color(c_linked)
        parts['cquantiles'].set_color(c_linked)
        parts['cmins'].set_color(c_linked)
        parts['cmaxes'].set_color(c_linked)
        parts['cbars'].set_color(c_linked)

        # set up labels ect
        ax.set_title(f'{site}-{mode}')
        if i == len(mode_sites) - 1:
            ax.set_xlabel('Month')
            ax.set_xticks(np.arange(1, len(plot_months) * nyears + 1) + 0.25)
            ax.set_xticklabels(mks)
            ax.set_xlim(0.5, len(mks) + 0.5)
        if i == len(mode_sites) // 2:
            ax.set_ylabel('kg DM/ha/day')

        ax.legend(handles=[
            Patch(facecolor=c_non_linked, label='Unlinked Storyline'),
            Patch(facecolor=c_linked, label='Linked Storyline'),
        ])

    pg_fig.suptitle(sup_title)
    pg_fig.tight_layout()
    if show:
        plt.show()
    pg_fig.savefig(outpath)


if __name__ == '__main__':
    plot_multi_year_monthly(
        outpath=r"D:\mh_unbacked\SLMACC_2020_norm\temp_storyline_files\test_multi\test_plot.png",
        mode_sites=default_mode_sites,
        impact_data=pd.read_csv(r"D:\mh_unbacked\SLMACC_2020_norm\temp_storyline_files\test_multi\id_probs.csv"),
        nyears=3,
        sup_title='test_plot',
        show=True
    )
