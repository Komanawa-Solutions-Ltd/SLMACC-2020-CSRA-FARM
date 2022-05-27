"""
created matt_dumont 
on: 27/05/22
"""
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
import ksl_env
from Storylines.storyline_building_support import default_mode_sites
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite
from Storylines.irrigation_mapper import get_irr_by_quantile
from BS_work.IID.IID import run_IID
from Storylines.storyline_evaluation.storyline_eval_support import calc_cumulative_impact_prob
from Ecological_flows.v2.alternate_restrictions import new_flows, make_new_rest_record, naturalise_historical_flow, \
    alternate_rest_dir
from Storylines.storyline_runs.run_random_suite import generate_random_suite


def recalc_story_prob(storyline_dict, new_rests):
    """
    recalcs the storyline probablities for all the new flow regimes
    :param storyline_dict:
    :param new_rests:
    :return:
    """
    assert isinstance(storyline_dict, dict)
    # run the storyline prob without irrigation restrictions
    outdata = run_IID(storyline_dict, add_irr_prob=False)
    outdata.rename({"log10_prob": "log10_prob_weather"}, inplace=True)
    outdata.set_index("ID")

    # get rest mappers
    rest_mappers = {'base': get_irr_by_quantile()}

    for name in new_rests:
        irr_quantile_dir = os.path.join(alternate_rest_dir, f'{name}-rest_mapper')
        detrend_rest = os.path.join(alternate_rest_dir, f'{name}-detrend_restriction_record.csv')
        if not os.path.exists(detrend_rest) or not os.path.exists(irr_quantile_dir):
            raise ValueError(f'detrending or percentile mapper missing for: {name}')
        rest_mappers[name] = get_irr_by_quantile(recalc=True, outdir=irr_quantile_dir, rest_path=detrend_rest)

    for k, sl in storyline_dict.items():
        # current_prob
        prob = sl.loc[:, 'rest_per'].values()
        prob[prob > 0] = np.log10(0.5 - np.abs(0.5 - prob[prob > 0]))
        outdata.loc[k, 'base_rest_prob'] = prob.sum()

        # do not need to map current prob to fraction rest (already done in the storyline building

        # map fraction rest to new prob for each new restriction name
        for name, mapper in new_rests:
            prob = map_storyline_frac_to_prob(sl, rest_mapper=rest_mappers[name])
            prob[prob > 0] = np.log10(0.5 - np.abs(0.5 - prob[prob > 0]))
            outdata.loc[k, f'{name}_rest_prob'] = prob.sum()

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


def export_cum_percentile(prob_data, nyr, outdir, step_size=0.1):
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
    for mode, site in default_mode_sites:
        if mode == 'dryland':
            continue

        if nyr == 1:
            data = get_1yr_data(bad_irr=True, good_irr=True)
        else:
            print('reading data')
            data = get_nyr_suite(nyr, site=site, mode=mode)
        # todo integrate prob_data to data via ID (need to add irrigation status to ID of data)
        # todo check that dryland prob and irrigated prob match.

        data.dropna(inplace=True)
        y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000

        prob = np.round(np.arange(0.01, 1, 0.01), 2)
        outdata = pd.DataFrame(index=prob)
        outdata.index.name = 'probability'

        for name in list(new_flows.keys()) + ['base']:
            # todo need to get the new probabilities (or all probablities)
            x = None  # todo place holder

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


def plot_export_prob_change_stats(prob_data):  # todo what do I want to do here?
    raise NotImplementedError


def main(recalc=False):
    base_outdir = os.path.join(ksl_env.slmmac_dir, 'eco_modelling', 'random')

    hdf_path = os.path.join(base_outdir, 'random_probs.hdf')
    if os.path.exists(hdf_path) and not recalc:
        data = pd.read_hdf(hdf_path, 'random')
    else:
        # make the storyline dictionary (pull from randomn
        sl_dict = {}
        n = 70000
        storylines = generate_random_suite(n, use_default_seed=True, save=False, return_story=True,
                                           bad_irr=True)
        sl_dict.update({f'rsl-{k:06d}-bad': v for k, v in enumerate(storylines)})
        storylines = generate_random_suite(n, use_default_seed=True, save=False, return_story=True,
                                           bad_irr=False)
        sl_dict.update({f'rsl-{k:06d}-good': v for k, v in enumerate(storylines)})
        # todo spot check the storylines to ensure they are the same (they should be), probably best to read in
        # a handful of random saved storylines on dickie and use them.

        data = recalc_story_prob(sl_dict, list(new_flows.keys()))
        data.to_csv(os.path.join(base_outdir, 'random_probs.csv'))
        data.to_hdf(hdf_path, 'random')

    # plot and export percentile changes in probability for different scenarios.
    plot_export_prob_change_stats(data)

    # export exceedence probs
    poss_yrs = [1, 2, 3, 5, 10]
    for y in poss_yrs:
        export_cum_percentile(data, y, os.path.join(base_outdir, 'exceedence', f'exceedence_{y}yr'))
