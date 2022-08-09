"""
created matt_dumont 
on: 22/07/22
"""
import netCDF4 as nc
import numpy as np
from pathlib import Path
import zipfile
from Storylines.storyline_runs.run_random_suite import get_mean_1yr_data, default_mode_sites

month_len = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def package_random(outdir, zippath, bad_irr):
    """

    :param nc_files: lsit of paths
    :param irr_type: list of irr type (bad, good)
    :param irr_mode: either dryland or irrigated (incl storage)
    :return:
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    mean_data = get_mean_1yr_data(correct=False)
    mean_data.loc[:, 'kid'] = mean_data.loc[:, 'ID'] + '_' + mean_data.loc[:, 'irr_type'] + '_irr'
    mean_data.set_index('kid', inplace=True)
    with zipfile.ZipFile(zippath) as zfile:
        all_names = zfile.namelist()
        for mode, site in default_mode_sites:
            if mode == 'dryland':
                use_mode = 'dryland'
            else:
                use_mode = 'irrigated'

            if bad_irr:
                irr = 'bad'
            else:
                irr = 'good'

            use_file_list = [e for e in all_names if f'{site}-{mode}' in e]

            pg_data = []
            sids = []
            probs = []
            out_irr = []
            expect_months = np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6])
            for i, n in enumerate(use_file_list):
                if i % 100 == 0:
                    print(f'{i} of {len(use_file_list)}')
                sid = Path(n)
                sid_mean = '-'.join(sid.name.split('-')[0:2]) + f'_{irr}_irr'
                sid = int(sid.name.split('-')[1])
                with zfile.open(n) as f:
                    with nc.Dataset('dummy', mode='r', memory=f.read()) as data:
                        assert data.dimensions['sim_month'].size == 12
                        assert (np.array(data.variables['m_month']) == expect_months).all()
                        key = 'm_PGR'
                        temp = np.array(data.variables[key]).transpose() * np.array(
                            [month_len[e] for e in expect_months])
                        temp = np.concatenate((temp, temp.sum(axis=1)[:, np.newaxis]), axis=1)
                        pg_data.append(temp.round())
                        sids.extend(np.repeat(sid, len(temp)))
                        out_irr.extend(np.repeat('bad' in irr, len(temp)))
                        probs.extend(np.repeat(mean_data.loc[sid_mean, f'log10_prob_{use_mode}'], len(temp)))
            pg_data = np.concatenate(pg_data)

            sids = np.array(sids)
            probs = np.array(probs)
            out_irr = np.zeros(probs.shape) + bad_irr
            columns = (['storyline', 'bad_irr'] + [f'pg_{m:02d}' for m in expect_months]
                       + ['pg_1yr', f'log10_prob_{use_mode}'])
            outdata = np.concatenate((sids[:, np.newaxis], out_irr[:, np.newaxis],
                                      pg_data, probs[:, np.newaxis]), axis=1)
            outdata = outdata.transpose()

            # save as np.z file
            default_type = np.uint16
            use_types = {c: default_type for c in columns}
            use_types.update({'bad_irr': np.bool_,
                              'storyline': np.uint32,
                              f'log10_prob_{use_mode}': np.float})
            kwargs = {}
            for a, k in zip(outdata, columns):
                kwargs[k] = a.astype(use_types[k])
            outpath = outdir.joinpath(f'{site}-{mode}_{irr}.npz')
            np.savez_compressed(outpath, **kwargs)


# todo need to re-run probability across full suite otherwise it is wrong.
# todo need to update the get data to get mean data and get full dataset, both in this repo and in the final repo
# todo re-run nyr from full suite

# todo review below to decide what needs to be done with new datasets
# #### model runs ####
# Storylines/storyline_runs/historical_quantified_1yr_detrend.py # have run
# Storylines\storyline_runs\historical_quantified_1yr_trend.py # have run

# Storylines/storyline_runs/run_random_suite.py # have run
# Storylines/storyline_runs/run_nyr_suite.py # have run


# #### exports and plots ####
# Storylines/storyline_evaluation/plot_historical_trended.py # have run
# Storylines/storyline_evaluation/plot_historical_detrended.py # have run

# Storylines/storyline_evaluation/export_cum_percentile.py # have run
# Storylines\storyline_evaluation\plot_site_v_site.py # have run
# Storylines/storyline_evaluation/plots.py # have run
# Storylines/storyline_evaluation/storyline_slection/stories_to_ws_pg_threshold.py # have run
# Storylines/storyline_evaluation/plot_cumulative_historical_v_modelled.py # run
# Storylines\final_plots\prob_and_pg_with_storage.py # have run


if __name__ == '__main__':
    # todo do bad irr, save these file somewhere
    package_random(Path.home().joinpath('Downloads/full_suite'), '/home/matt_dumont/Downloads/random_bad_irr.zip',
                   bad_irr=True)
