"""
created matt_dumont 
on: 19/07/22
"""
import numpy as np
import pandas as pd
from pathlib import Path
from project_base import proj_root
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from Storylines.storyline_evaluation.storyline_characteristics_for_impact import get_suite
from Storylines.storyline_runs.run_random_suite import get_1yr_data

base_site_mode = (
    'eyrewell-irrigated',
    'oxford-irrigated')

use_mode_sites = (
    ('dryland', 'oxford'),
    ('irrigated', 'eyrewell'),
    ('irrigated', 'oxford'),
)


def get_monthly_scen_data(scenario):
    base_path = Path(proj_root)
    if scenario == 'baseline':
        ext = 'Storylines/final_storylines/scenario_pg_data/baseline-raw.csv.zip'
    elif scenario == 'scare':
        ext = 'Storylines/final_storylines/scenario_pg_data/scare-raw.csv'
    elif scenario == 'hurt':
        ext = 'Storylines/final_storylines/scenario_pg_data/hurt-raw.csv'
    else:
        raise ValueError(f'unexpected value for scenario: {scenario}')

    data = pd.read_csv(base_path.joinpath(ext)).mean()
    return data


def get_limits(scenario, tolerance):
    """
    make monthly_limits: None or monthly limits (kg dm/ha/month) format is {'site'-'mode':
                                                                                {month(int): (min, max)}
                                                                                }


    :param scenario:
    :param tolerance: fraction
    :return:
    """
    monthly = get_monthly_scen_data(scenario)
    out = {}
    for sm in base_site_mode:
        if 'dryland' in sm:
            continue
        temp = {}
        for m in range(1, 13):
            if m in [6, 7, 8]:
                continue
            t = monthly[sm][m]
            use_tol = t * tolerance
            if use_tol < 5:
                use_tol = 10 * 30
            temp[m] = (t - use_tol, t + use_tol)
        out[sm] = temp
    return out


def get_colors(vals, cmap_name='tab20'):
    n_scens = len(vals)
    if n_scens < 20:
        cmap = get_cmap(cmap_name)
        colors = [cmap(e / (n_scens + 1)) for e in range(n_scens)]
    else:
        colors = []
        i = 0
        cmap = get_cmap(cmap_name)
        for v in vals:
            colors.append(cmap(i / 20))
            i += 1
            if i == 20:
                i = 0
    return colors


def get_indicative_stories(scenario):
    scen_data = get_monthly_scen_data(scenario)

    data = get_1yr_data(bad_irr=True, good_irr=True, correct=False)
    data.loc[:, 'kid'] = data.loc[:, 'ID'] + data.loc[:, 'irr_type']
    data = data.set_index('kid')
    use_keys = []
    for sm in base_site_mode:
        for m in range(1, 13):
            if m in [6, 7, 8]:
                continue
            use_keys.append(f'{sm}_pg_m{m:02d}')

    data = data.dropna()
    dif = (data - scen_data).loc[:, use_keys].abs().sum(axis=1)
    data.loc[:, 'dif'] = dif
    dif = dif.sort_values()
    # plot data for visual inspection
    fig, axs = plt.subplots(2)
    use_data = dif.index[0:10]
    plot_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    for sm, ax in zip(base_site_mode, axs):
        for d, c in zip(use_data, get_colors(use_data)):
            ax.plot(range(12), data.loc[d, [f'{sm}_pg_m{m:02d}' for m in plot_months]], label=d, c=c, ls=':')
        ax.set_title(sm)
        ax.plot(range(12), scen_data.loc[[f'{sm}_pg_m{m:02d}' for m in plot_months]], label='base', c='deeppink')
    ax.legend()
    print(data.loc[use_data, ['log10_prob_irrigated', 'dif']])
    fig, axs = plt.subplots(2)
    fig.suptitle('top picks')
    top_picks = ['rsl-031008good', 'rsl-061803bad']
    for sm, ax in zip(base_site_mode, axs):
        for d, c in zip(top_picks, get_colors(use_data)):
            ax.plot(range(12), data.loc[d, [f'{sm}_pg_m{m:02d}' for m in plot_months]], label=d, c=c, ls=':')
        ax.set_title(sm)
        ax.plot(range(12), scen_data.loc[[f'{sm}_pg_m{m:02d}' for m in plot_months]], label='base', c='deeppink')
    ax.legend()

    plt.show()

    pass


best_fit_stories = {  # minimum value
    'hurt': 'rsl-057004bad',
    'scare': 'rsl-061803bad',
    'baseline': 'rsl-032586bad',
}

from zipfile import ZipFile


def get_storylines(ids, irr_types):
    """
    get the storylines for a given dataset
    :param ids: list of storyline ids (e.g., 'rsl-069998'])
    :param irr_types: a list of irr types one of:
                        * 'bad_irr': where irrigation restrictions range from 50th to 99th percentile
                        * 'good_irr': where irrigation restrictions range from 1st to 50th percentile

    :return:
    """
    outdata = []
    base_data_path = Path(proj_root).joinpath('Storylines')
    assert len(ids) == len(irr_types), 'expected ids and irr_types to be the same length'
    for idd, itype in zip(ids, irr_types):
        if itype == 'good_irr':
            with ZipFile(base_data_path.joinpath('random_good_irr.zip')) as zf:
                with zf.open(f'random_good_irr/{idd}.csv') as f:
                    t = pd.read_csv(f)
            outdata.append(t)
        elif itype == 'bad_irr':
            with ZipFile(base_data_path.joinpath('random_bad_irr.zip')) as zf:
                with zf.open(f'random_bad_irr/{idd}.csv') as f:
                    t = pd.read_csv(f)
            outdata.append(t)
        else:
            raise ValueError(f'unexpected value in irr_types: {itype} expected ["good_irr" or "bad_irr"')
    return outdata


from Climate_Shocks.Stochastic_Weather_Generator.irrigation_flow_generator import get_irrigation_generator, month_len


def get_best_fit_storyline_data():
    irr_gen = get_irrigation_generator()
    outdir = Path(proj_root).joinpath('Storylines/final_storylines/scenario_pg_data/best_fit_story')
    data = get_1yr_data(bad_irr=True, good_irr=True, correct=False)
    data.loc[:, 'kid'] = data.loc[:, 'ID'] + data.loc[:, 'irr_type']
    data = data.set_index('kid')
    for sn, sl in best_fit_stories.items():
        print(sn)
        # storyline,
        sl_data = get_storylines([sl.strip('bad')], ['bad_irr'])[0]
        sl_data.to_csv(outdir.joinpath(f'{sn}_best_fit_{sl}_story.csv'))

        # impact_data,
        data.loc[sl].to_csv(outdir.joinpath(f'{sn}_best_fit_{sl}_impact.csv'))

        #  flow data...
        sims = 10
        outdata_shp = (sims, sum([month_len[m] for m in sl_data.month]))
        outdata = np.zeros(outdata_shp) * np.nan
        start = 0
        for i, d in sl_data.iterrows():
            m = d.month
            if d.precip_class == 'D':
                p = 'D'
            else:
                p = 'ND'

            end = start + month_len[m]
            if m in [5, 6, 7, 8]:
                outdata[:, start:end] = np.nan
            else:
                temp = irr_gen.get_data(sims, key=f'm{m:02d}-{p}',suffix='rest', suffix_selection=d.rest,tolerance=0.01)
                outdata[:, start:end] = temp
            start = end

        idx = pd.date_range('2025-07-01', '2026-06-30')
        outdata = pd.DataFrame(outdata.transpose(), index=idx, columns=[f'sim_{e:02d}' for e in range(sims)])
        outdata.to_csv(outdir.joinpath(f'{sn}_best_fit_{sl}_flow_data.csv'))



if __name__ == '__main__':
    get_best_fit_storyline_data()  # todo debug, see what happens
