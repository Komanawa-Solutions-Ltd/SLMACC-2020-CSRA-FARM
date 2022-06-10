"""
created matt_dumont
on: 27/05/22
"""
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
import ksl_env
from Storylines.storyline_building_support import default_mode_sites
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite, get_nyr_idxs
from Storylines.irrigation_mapper import get_irr_by_quantile
from BS_work.IID.IID import run_IID
from Storylines.storyline_evaluation.storyline_eval_support import calc_cumulative_impact_prob
from Ecological_flows.v2.alternate_restrictions import new_flows, make_new_rest_record, naturalise_historical_flow, \
    alternate_rest_dir
from Storylines.storyline_runs.run_random_suite import generate_random_suite
from pathlib import Path

base_outdir = os.path.join(ksl_env.slmmac_dir, 'eco_modelling', 'random')
fig_size = (10, 8)  # todo
os.makedirs(base_outdir, exist_ok=True)


def recalc_story_prob(storyline_dict, new_rests):
    """
    recalcs the storyline probablities for all the new flow regimes
    :param storyline_dict:
    :param new_rests:
    :return:
    """
    print('recalculating storyline probs')
    assert isinstance(storyline_dict, dict)
    # run the storyline prob without irrigation restrictions
    outdata = run_IID(storyline_dict, add_irr_prob=False)
    outdata.rename({"log10_prob": "log10_prob_weather"}, axis=1, inplace=True)
    outdata.set_index("ID", inplace=True)

    # get rest mappers
    rest_mappers = {'base': get_irr_by_quantile()}

    for name in new_rests:
        irr_quantile_dir = os.path.join(alternate_rest_dir, f'{name}-rest_mapper')
        detrend_rest = os.path.join(alternate_rest_dir, f'{name}-detrend_restriction_record.csv')
        if not os.path.exists(detrend_rest) or not os.path.exists(irr_quantile_dir):
            raise ValueError(f'detrending or percentile mapper missing for: {name}')
        rest_mappers[name] = get_irr_by_quantile(recalc=True, outdir=irr_quantile_dir, rest_path=detrend_rest)

    for i, (k, sl) in enumerate(storyline_dict.items()):
        if i % 1000 == 0:
            print(f'recalculating new rest prob for {i}: {k}')
        # current_prob
        prob = sl.loc[:, 'rest_per'].values
        prob[prob > 0] = np.log10(0.5 - np.abs(0.5 - prob[prob > 0]))
        outdata.loc[k, 'base_rest_prob'] = prob.sum()

        # do not need to map current prob to fraction rest (already done in the storyline building

        # map fraction rest to new prob for each new restriction name
        for name in new_rests:
            prob = map_storyline_frac_to_prob(sl, rest_mapper=rest_mappers[name])
            prob[prob > 0] = np.log10(0.5 - np.abs(0.5 - prob[prob > 0]))
            outdata.loc[k, f'{name}_rest_prob'] = prob.sum()

    for name in list(new_rests) + ['base']:
        outdata.loc[:, f'{name}_rest_prob'] += outdata.loc[:, "log10_prob_weather"]

    return outdata


def map_storyline_frac_to_prob(story, rest_mapper):
    """
    maps the fraction of restrictions to a probability under a given rest mapper
    :param story:
    :param rest_mapper:
    :return:
    """
    out = []
    story.loc[:, 'precip_class_prev'] = story.loc[:, 'precip_class'].shift(1).fillna('A')
    iter_keys = ['month', 'rest', 'precip_class', 'precip_class_prev']
    for m, r, p, pp in story.loc[:, iter_keys].itertuples(False, None):
        if m in (5, 6, 7, 8,):
            out.append(0)
        else:
            key = f'{p}-{pp}'.replace('W', 'ND').replace('A', 'ND')
            t2 = rest_mapper[key]
            t = np.interp(r, t2.loc[:, m], t2.index)
            out.append(t)

    return np.array(out)


def export_cum_percentile(prob_data, nyr, outdir, step_size=0.1, recalc=False):
    """
    exports the cumulative probabilty for the new irrigation mappers (and the base value)
    :param nyr:
    :param outdir:
    :param step_size:
    :return:
    """
    prob_data = prob_data.copy(deep=True)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out = {}
    for mode, site in default_mode_sites:
        hdf_path = os.path.join(outdir, f'{site}-{mode}_new_rest_cumulative_prob.hdf')
        if os.path.exists(hdf_path) and not recalc:
            data = pd.read_hdf(hdf_path, 'random')
            out[(site, mode)] = data
            continue

        if mode == 'dryland':
            continue

        if nyr == 1:
            data = get_1yr_lines(prob_data)

        else:
            print('reading data')
            data = get_nyr_lines(prob_data=prob_data, nyr=nyr, site=site, mode=mode, recalc=recalc)

        data.dropna(inplace=True)
        y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000

        prob = np.round(np.arange(0.01, 1, 0.01), 2)
        outdata = pd.DataFrame(index=prob)
        outdata.index.name = 'probability'

        for name in list(new_flows.keys()) + ['base']:
            x = data.loc[:, f'{name}_rest_prob']

            cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=y,
                                                            prob=x, stepsize=step_size,
                                                            more_production_than=False)
            f = interp1d(cum_prob, cum_pgr)
            outdata.loc[:, f'{name}_non-exceedance_pg'] = f(prob)

            cum_pgr2, cum_prob2 = calc_cumulative_impact_prob(pgr=y,
                                                              prob=x, stepsize=step_size,
                                                              more_production_than=True)
            f = interp1d(cum_prob2, cum_pgr2)
            outdata.loc[:, f'{name}_exceedance_pg'] = f(prob)

        outdata.to_csv(os.path.join(outdir, f'{site}-{mode}_new_rest_cumulative_prob.csv'))
        outdata.to_hdf(hdf_path, 'random')
        out[(site, mode)] = outdata

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


def plot_export_prob_change_stats(prob_data):
    data = get_1yr_lines(prob_data)
    # todo do I need density?
    # plot new vs old prob
    for name in list(new_flows.keys()):
        fig, ax = plt.subplots(figsize=fig_size)
        y = data.loc[:, f'{name}_rest_prob'] - data.loc[:, "log10_prob_weather"]
        x = data.loc[:, f'base_rest_prob'] - data.loc[:, "log10_prob_weather"]
        ax.scatter(x, y)
        ax.set_ylable(f'{name}_rest_prob')
        ax.set_xlable(f'base_rest_prob')

    # todo plot pg (xaxsis) vs prob(new)/prob(base)
    for mode, site in default_mode_sites:
        if mode == 'dryland':
            continue
        fig, ax = plt.subplots(figsize=fig_size)
        x = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000
        colors = get_colors(new_flows.keys())

        for c, name in zip(list(new_flows.keys()), colors):
            # recall that probablities are log10(prob)
            y = 10 ** ((data.loc[:, f'{name}_rest_prob'] - data.loc[:, "log10_prob_weather"])
                       - (data.loc[:, f'base_rest_prob'] - data.loc[:, "log10_prob_weather"]))
            ax.scatter(x, y, label=name, c=c)
        ax.set_title(f'{site} - {mode}')
        ax.set_ylable(f'scenario rest prob / base rest prob')
        ax.set_xlable(f'Pasture growth (Tons DM/yr)')
        ax.legend()
    # todo what do I want to do here?, probably need to see the data
    # todo save figs
    raise NotImplementedError


def get_1yr_lines(prob_data):
    data = get_1yr_data(bad_irr=True, good_irr=True)
    data.loc[:, 'ID'] = data.loc[:, 'ID'] + '-' + data.loc[:, 'irr_type']
    data.set_index('ID', inplace=True)
    data = pd.merge(data, prob_data, left_index=True, right_index=True)
    # join weather and irrigaiton probs

    assert np.isclose(data.loc[:, 'log10_prob_irrigated'],
                      data.loc[:, 'base_rest_prob']).all(), 'irrigated probabilities do not match'
    assert np.isclose(data.loc[:, 'log10_prob_dryland'],
                      data.loc[:, "log10_prob_weather"]).all(), 'dry land probabilities do not match'
    return data


def get_nyr_lines(prob_data, nyr, site, mode, recalc=False):
    nyr_data = get_nyr_suite(nyr=nyr, site=site, mode=mode).loc[:, [f'{site}-{mode}_pg_yr{nyr}']]
    nyr_idxs = get_nyr_idxs(nyr=nyr, mode=mode)
    hdf_path = os.path.join(base_outdir, f'{nyr}_yr_{site}_{mode}_random_probs.hdf')
    if os.path.exists(hdf_path) and not recalc:
        data = pd.read_hdf(hdf_path, 'random')
        return data
    else:
        for name in list(new_flows.keys()) + ['base']:
            key = f'{name}_rest_prob'
            probs = prob_data.loc[nyr_idxs.flatten(), key].values.reshape(nyr_idxs.shape)
            nyr_data.loc[:, key] = probs.sum(axis=1)
        nyr_data.to_hdf(hdf_path, 'random')
    return nyr_data


def plot_exceedence_prob(nyr, data):
    for mode, site in default_mode_sites:
        if mode == 'dryland':
            continue
        cum_data = data[(site, mode)]
        names = ['base'] + list(new_flows.keys())
        colors = get_colors(names)
        fig, ax = plt.subplots(figsize=fig_size)

        # cum_data keys are f'{name}_exceedance_pg' or f'{name}_non-exceedance_pg'

        for k, c in zip(names, colors):
            ax.plot(cum_data.loc[:, f'{k}_exceedance_pg'], cum_data.index, c=c, label=f'{k}')
        ax.legend()
        ax.set_title('exceedence probability')
        ax.set_yable('probability')
        ax.set_xable(f'pasture growth tons/{nyr} y')

        # todo plot, save

    raise NotImplementedError


def main(recalc=False, plot=False):
    hdf_path = os.path.join(base_outdir, 'random_probs.hdf')
    if os.path.exists(hdf_path) and not recalc:
        data = pd.read_hdf(hdf_path, 'random')
    else:
        # make the storyline dictionary (pull from randomn
        sl_dict = {}
        n = 70000

        # need to read these in from the saved set... seed not working...

        bad_dir = Path(r'D:\mh_unbacked\SLMACC_2020_norm\temp_storyline_files\random_bad_irr')

        good_dir = Path(r'D:\mh_unbacked\SLMACC_2020_norm\temp_storyline_files\random_good_irr')

        for gn, gd in zip(['good', 'bad'], [good_dir, bad_dir]):
            paths = gd.glob('*.csv')
            for i, p in enumerate(paths):

                if i % 1000 == 0:
                    print(f'reading {i} for {gn}')
                temp = pd.read_csv(p, index_col=0)
                t = {'rest': 'float64',
                     'rest_per': 'float64',
                     'year': 'int64',
                     'month': 'int64'}
                for e, v in t.items():
                    temp.loc[:, e] = temp.loc[:, e].astype(v)
                sl_dict[p.name.replace('.csv', f'-{gn}')] = temp

        data = recalc_story_prob(sl_dict, list(new_flows.keys()))
        data.to_csv(os.path.join(base_outdir, 'random_probs.csv'))
        data.to_hdf(hdf_path, 'random')

    if plot:
        # plot and export percentile changes in probability for different scenarios.
        plot_export_prob_change_stats(data)

    # export exceedence probs
    poss_yrs = [1, 2, 3, 5, 10]
    for y in poss_yrs:
        cum_data = export_cum_percentile(data, y, os.path.join(base_outdir, 'exceedence', f'exceedence_{y}yr'),
                                         recalc=recalc)
        if plot:
            plot_exceedence_prob(y, cum_data)


if __name__ == '__main__':
    main(recalc=True, plot=False)  # todo plot
