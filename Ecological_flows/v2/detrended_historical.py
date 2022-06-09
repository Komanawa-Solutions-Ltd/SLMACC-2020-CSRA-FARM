"""
created matt_dumont 
on: 27/05/22
"""
import itertools

import ksl_env
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from Climate_Shocks.get_past_record import get_vcsn_record
from Pasture_Growth_Modelling.basgra_parameter_sets import get_params_doy_irr, create_days_harvest, \
    create_matrix_weather
from Pasture_Growth_Modelling.calculate_pasture_growth import calc_pasture_growth, calc_pasture_growth_anomaly
from Pasture_Growth_Modelling.basgra_parameter_sets import default_mode_sites
from Ecological_flows.v2.alternate_restrictions import new_flows, get_new_flow_rest_record
from Storylines.storyline_evaluation.storyline_eval_support import calc_cumulative_impact_prob

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from basgra_python import run_basgra_nz

base_outdir = Path(ksl_env.slmmac_dir).joinpath('eco_modelling', 'historical_detrended')
base_outdir.mkdir(exist_ok=True, parents=True)
figsize = (16.5, 9.25)


def run_past_basgra_irrigated(site='eyrewell', reseed=True, version='trended', mode='irrigated',
                              flow_version='base'):
    print(f'running: {mode}, {site}, flow: {flow_version}, reseed: {reseed}')

    if version == 'detrended':
        v = 'detrended2'
    else:
        v = version
    weather = get_vcsn_record(version=v, site=site)
    rest = get_new_flow_rest_record(name=flow_version, version=version)
    params, doy_irr = get_params_doy_irr(mode)
    all_out = []
    for y in range(1972, 2019):
        temp_weather = weather.loc[(weather.index >= f'{y}-07-01') & (weather.index < f'{y + 1}-07-01')]
        temp_rest = rest.loc[(rest.index >= f'{y}-07-01') & (rest.index < f'{y + 1}-07-01')]

        matrix_weather = create_matrix_weather(mode, temp_weather, temp_rest, fix_leap=False)
        days_harvest = create_days_harvest(mode, matrix_weather, site, fix_leap=False)

        if not reseed:
            days_harvest.loc[:, 'reseed_trig'] = -1

        out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
        out.loc[:, 'per_PAW'] = out.loc[:, 'PAW'] / out.loc[:, 'MXPAW']

        pg = pd.DataFrame(calc_pasture_growth(out, days_harvest, mode='from_yield', resamp_fun='mean', freq='1d'))
        out.loc[:, 'pg'] = pg.loc[:, 'pg']
        out.loc[:, 'season'] = y
        all_out.append(out)

    all_out = pd.concat(all_out)
    all_out = calc_pasture_growth_anomaly(all_out, fun='mean')

    return all_out


def get_run_basgra_for_historical_new_flows(version, recalc=False):
    """

    :param version: tended or detrended
    :param recalc:
    :return:
    """
    pickle_path = base_outdir.joinpath('pickles', f'new_flow_basgra_{version}.p')
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists() and not recalc:
        out = pickle.load(pickle_path.open('rb'))
        return out

    out = {}
    names = ['base'] + list(new_flows.keys())
    for mode, site in default_mode_sites:
        if version == 'detrended' and site == 'oxford':
            continue  # cannot run detreneded for oxford
        if mode == 'dryland':
            continue
        for name in names:
            temp = run_past_basgra_irrigated(site=site, reseed=True, version=version, mode=mode,
                                             flow_version=name)
            out[(mode, site, name)] = temp

    pickle.dump(out, pickle_path.open('wb'))
    return out


def get_make_cumulative_prob_historical(nyr, site, mode, flow_name, version, sequential, recalc=False,
                                        recalc_basgra=False):
    pickle_path = base_outdir.joinpath('pickles', f'new_flow_cumulative_{site}-{mode}-{version}-{nyr}-{sequential}.hdf')
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists() and not recalc:
        out = pd.read_hdf(pickle_path, 'historical')
        return out
    print(f'making cumulative prob for: nyr={nyr}, site={site}, mode={mode}, '
          f'flow_name={flow_name}, version={version}, sequential={sequential}')
    data = get_run_basgra_for_historical_new_flows(version, recalc=recalc_basgra)
    pgr_key = 'pg'
    temp_data = data[(mode, site, flow_name)]
    assert isinstance(temp_data, pd.DataFrame)
    temp_data = temp_data.groupby('season').sum()
    temp_data.loc[:, pgr_key] *= 1 / 1000  # change unit to tons
    if nyr == 1:
        pgr = temp_data.loc[:, pgr_key]
        prob = np.zeros(pgr.shape) + 1
    else:
        if sequential:
            pgr = temp_data.rolling(nyr).sum().loc[:, pgr_key]
            pgr = pgr[np.isfinite(pgr)]
            prob = np.zeros(pgr.shape) + 1
        else:
            idxs = np.random.randint(0, len(temp_data), 10000)
            pgr = temp_data.iloc[idxs].rolling(nyr).sum().loc[:, pgr_key]
            pgr = pgr[np.isfinite(pgr)]

            prob = np.zeros(pgr.shape) + 1

    cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=pgr,
                                                    prob=prob, stepsize=0.1,
                                                    more_production_than=False)
    out = pd.DataFrame({'prob': cum_prob, 'pgr_non_exceed': cum_pgr})
    out.to_hdf(pickle_path, 'historical')
    return out


def plot_resampling_new_flow(nyr, site, mode):  # todo cumulative prob
    # todo cannot do this until I have the alternate restirctions
    # todo reference Storylines/storyline_evaluation/plot_historical_detrended.py
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize)

    # todo all data

    # todo per version per sequetntila

    # sequential vs non-sequential

    raise NotImplementedError


def main(recalc=False, plot=False):
    for v in ['trended', 'detrended']:
        data = get_run_basgra_for_historical_new_flows(version=v, recalc=recalc)

        for mode, site in default_mode_sites:
            if site == 'oxford' and v == 'detrended':
                continue
            if mode == 'dryland':
                continue
            years = [1, 2, 3, 5, 10]
            seqentials = [True, False]
            flow_names = list(new_flows.keys()) + ['base']

            # make the cumulative data
            for y, sq, fn in itertools.product(years, seqentials, flow_names):
                get_make_cumulative_prob_historical(nyr=y, site=site, mode=mode, flow_name=fn, version=v, sequential=sq,
                                                    recalc=recalc)

            if plot:
                # plot the cumulative data
                for y in years:
                    plot_resampling_new_flow(nyr=y, site=site, mode=mode)


if __name__ == '__main__':
    main(recalc=False, plot=False) # todo plot
