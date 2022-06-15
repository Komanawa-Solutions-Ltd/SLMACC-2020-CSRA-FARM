"""
created matt_dumont 
on: 15/06/22
"""
import matplotlib.pyplot as plt

from Ecological_flows.v2.alternate_restrictions import naturalise_historical_flow, make_new_rest_record, new_flows
from Ecological_flows.v2.detrended_historical import get_run_basgra_for_historical_new_flows, default_mode_sites
import pandas as pd
import numpy as np
from pathlib import Path
import ksl_env
import pickle

base_outdir = Path(ksl_env.slmmac_dir).joinpath('eco_modelling', 'historical_detrended')
base_outdir.mkdir(exist_ok=True, parents=True)
figsize = (16.5, 9.25)


def get_make_pg_flow_nat_daily_data(winter_takes=False, recalc=False):
    if winter_takes:
        extra = 'inc_wintertake'
    else:
        extra = 'exc_wintertake'

    pickle_path = base_outdir.joinpath('pickles', f'flow_and_pg_daily_{extra}.p')
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists() and not recalc:
        out = pickle.load(pickle_path.open('rb'))
        return out

    pg_data = get_run_basgra_for_historical_new_flows('trended')
    joint_data = {}
    for name in new_flows:
        nat = naturalise_historical_flow()
        data = make_new_rest_record(name, nat, winter_takes)
        data.loc[:, 'water_year'] = (data.index + pd.DateOffset(months=-6)).year
        data.loc[:, 'alf'] = data.loc[:, 'flow'].rolling(7).mean()

        for mode, site in default_mode_sites:
            if mode == 'dryland':
                continue
            pg = pg_data[(mode, site, name)]
            data.loc[:, f'{site}-{mode}'] = pg.loc[:, 'pg']  # todo is this correct?
        joint_data[name] = data
    pickle.dump(joint_data, pickle_path.open('wb'))
    return joint_data


def get_make_pg_flow_nat_annual_data(winter_takes=False, recalc=False):
    if winter_takes:
        extra = 'inc_wintertake'
    else:
        extra = 'exc_wintertake'

    pickle_path = base_outdir.joinpath('pickles', f'flow_and_pg_annual_{extra}.p')
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists() and not recalc:
        out = pickle.load(pickle_path.open('rb'))
        return out

    outdata = {}
    data = get_make_pg_flow_nat_daily_data(winter_takes=winter_takes, recalc=recalc)
    agg_dict = {
        'alf': 'min',
        'take': 'sum',

    }
    for mode, site in default_mode_sites:
        if mode == 'dryland':
            continue
        agg_dict[f'{site}-{mode}'] = 'sum'

    for k, v in data.items():
        assert isinstance(v, pd.DataFrame)
        temp = v.groupby('water_year').aggregate(agg_dict)
        outdata[k] = temp

    pickle.dump(outdata, pickle_path.open('wb'))
    return outdata


def get_make_resample_annual_data(nyrs, winter_takes=False, recalc=False):
    if winter_takes:
        extra = 'inc_wintertake'
    else:
        extra = 'exc_wintertake'

    pickle_path = base_outdir.joinpath('pickles', f'resample_{nyrs}_flow_and_pg_annual_{extra}.p')
    pickle_path.parent.mkdir(exist_ok=True)

    if pickle_path.exists() and not recalc:
        out = pickle.load(pickle_path.open('rb'))
        return out
    data = get_make_pg_flow_nat_annual_data(winter_takes=winter_takes, recalc=recalc)
    outdata = {}

    water_years = list(range(1972, 2019))
    num = 10000
    random_shape = (num, nyrs)
    np.random.seed(5654835)
    use_years = np.random.choice(water_years, (num, nyrs))

    agg_dict = {
        'alf': np.mean,
        'take': np.mean,

    }
    for mode, site in default_mode_sites:
        if mode == 'dryland':
            continue
        agg_dict[f'{site}-{mode}'] = np.sum

    for k, v in data.items():
        temp_out = pd.DataFrame(index=range(num), columns=agg_dict)
        for key, fun in agg_dict.items():
            temp = v.loc[use_years.flatten(), key].values.reshape(random_shape)
            temp_out.loc[:, key] = fun(temp, axis=1)
        outdata[k] = temp_out

    pickle.dump((use_years, outdata), pickle_path.open('wb'))
    return use_years, outdata


def get_daily_resample(idx, nyrs, winter_takes=False):
    """
    only use after running previous ones
    :param idx:
    :param nyrs:
    :param winter_takes:
    :return:
    """
    daily_data = get_make_pg_flow_nat_daily_data(winter_takes=winter_takes)
    use_years, resample = get_make_resample_annual_data(nyrs, winter_takes=winter_takes)

    outdata = {}
    for k, v in daily_data.items():
        out = []
        for y, uy in enumerate(use_years[idx]):
            temp = daily_data.loc[daily_data.water_year == uy].copy(deep=True)
            temp.loc[:, 'year'] = y
            out.append(temp)
        out = pd.concat(out).reset_index()
        outdata[k] = out
    return outdata


def plot_malf_v_pg(nyr, wintertakes=False):
    use_years, data = get_make_resample_annual_data(nyr, wintertakes)
    fig, axs = plt.subplots(2, 4, figsize=figsize, sharex=True, sharey=True)
    for ax, (k, v) in zip(axs.flatten(), data.items()):
        ax.scatter(v.alf, v.loc[:, 'eyrewell-irrigated'] / 1000)
        ax.set_title(k)
        ax.set_ylabel('PG growth')
        ax.set_xlabel('MALF')

    plt.show()


def one_to_one(ax, **kwargs):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    l = min(xmin, ymin)
    m = max(xmax, ymax)

    ax.plot([l, m], [l, m], **kwargs)


def plot_new_v_current(nyr, wintertakes=False):
    use_years, data = get_make_resample_annual_data(nyr, wintertakes)
    current = data.pop('current')

    for mode, site in default_mode_sites:
        if mode == 'dryland':
            continue
        key = f'{site}-{mode}'
        fig, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
        for ax, (k, v) in zip(axs.flatten(), data.items()):
            ax.scatter(current.loc[:, key] / 1000, v.loc[:, key] / 1000)
            ax.set_title(k)
            ax.set_ylabel('new')
            ax.set_xlabel('current')
            one_to_one(ax, ls=':')
        fig.suptitle(f'{key}: PG vs PG')

    fig, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    for ax, (k, v) in zip(axs.flatten(), data.items()):
        ax.scatter(current.alf, v.alf)
        ax.set_title(k)
        ax.set_ylabel('new')
        ax.set_xlabel('current')
        one_to_one(ax, ls=':')
    fig.suptitle('MALF vs MALF')

    plt.show()


# todo run, plot, and check!!!!
if __name__ == '__main__':
    recalc_data = False
    all_nyrs = [1, 3, 5, 10]
    if recalc_data:
        for y in all_nyrs:
            for v in [True, False]:
                print(f'recalcing data for yrs: {y} and winter_takes={v}')
                get_make_resample_annual_data(nyrs=y, winter_takes=v, recalc=recalc_data)
    plot_new_v_current(1)
