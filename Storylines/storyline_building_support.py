"""
 Author: Matt Hanson
 Created: 19/02/2021 3:38 PM
 """
import warnings
import pandas as pd
import os
import numpy as np
from Storylines.check_storyline import get_acceptable_events, get_past_event_frequency
from Climate_Shocks import climate_shocks_env
import itertools
import ksl_env
from scipy.interpolate import interp1d

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

irrig_season = (9, 10, 11, 12, 1, 2, 3, 4)

prev_month = {
    1: 12,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
}

month_fchange = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


# make all possible events
def make_sampling_options():
    acceptable = get_acceptable_events()
    outdata = {}
    for m in range(1, 13):
        # define array temp, precip, rest
        temp = []
        precip = []
        rest = []
        for k, v in acceptable.items():
            if m in v:
                temp.append(k.split('-')[0])
                precip.append(k.split('-')[1])
                rest.append(0)

        outdata[m] = np.array([temp, precip, rest]).transpose()

    return outdata


def map_storyline_rest(story):
    """
    assumes that the rest values are quantiles
    :param story:
    :return:
    """
    story.loc[:, 'precip_class_prev'] = story.loc[:, 'precip_class'].shift(1).fillna('A')
    story.loc[:, 'rest'] = [
        map_irrigation(m=m,
                       rest_quantile=rq,
                       precip=p,
                       prev_precip=pp) for m, rq, p, pp in story.loc[:, ['month', 'rest',
                                                                         'precip_class',
                                                                         'precip_class_prev']].itertuples(False, None)]


def map_irrigation(m, rest_quantile, precip, prev_precip):
    if m in (5, 6, 7, 8,):  # non irrigation months
        return 0
    key = f'{prev_precip}-{precip}'.replace('W', 'ND').replace('A', 'ND')
    rest_quantile = round(rest_quantile, 2)
    return round(_rest_data[key].loc[rest_quantile, m], 4)


def _get_irr_by_quantile(recalc=False):
    dnd = [['D', 'ND'], ['D', 'ND']]
    possible_quantiles = np.arange(1, 100) / 100

    rest_dir = os.path.join(climate_shocks_env.supporting_data_dir, 'rest_mapper')
    if not os.path.exists(rest_dir):
        os.makedirs(rest_dir)

    if os.listdir(rest_dir) == [f'{m1}-{m2}_rest.csv' for m1, m2 in itertools.product(*dnd)] and not recalc:
        out = {}
        for m1, m2 in itertools.product(*dnd):
            quantile = pd.read_csv(os.path.join(rest_dir, f'{m1}-{m2}_rest.csv'), index_col=0)
            quantile.columns = quantile.columns.astype(int)
            out[f'{m1}-{m2}'] = quantile
        return out

    rest_data = pd.read_csv(os.path.join(climate_shocks_env.supporting_data_dir,
                                         'restriction_record_detrend.csv'))
    rest_data = rest_data.groupby(['year', 'month']).mean()
    event_data = pd.read_csv(climate_shocks_env.event_def_path, skiprows=1)
    event_data = event_data.set_index(['year', 'month'])
    event_data.loc[:, 'rest'] = rest_data.loc[:, 'f_rest']
    event_data.loc[:, 'dnd'] = [fix_precip(e) for e in event_data.loc[:, 'precip']]
    event_data.loc[:, 'prev_dnd'] = [fix_precip(e) for e in event_data.loc[:, 'prev_precip']]

    out = {}
    for m1, m2 in itertools.product(*dnd):
        outdata = pd.DataFrame(index=possible_quantiles, columns=irrig_season)
        for m in irrig_season:
            temp = event_data.loc[:, m, :]
            temp = temp.loc[(temp.dnd == m2) & (temp.prev_dnd == m1)]
            outdata.loc[:, m] = temp.loc[:, 'rest'].quantile(possible_quantiles).fillna(0)
        out[f'{m1}-{m2}'] = outdata
        outdata.to_csv(os.path.join(rest_dir, f'{m1}-{m2}_rest.csv'))
    return out


_rest_data = _get_irr_by_quantile()


def fix_precip(x):
    if x == 1:
        return "D"
    else:
        if not np.isnan(x):
            return "ND"


base_rest_data = {
    1: 0.5,
    2: 0.5,
    3: 0.5,
    4: 0.5,
    9: 0.5,
    10: 0.5,
    11: 0.5,
    12: 0.5,

    # non-irrigation seasons
    5: 0.01,
    6: 0.01,
    7: 0.01,
    8: 0.01,
}

base_events = {e: ('A', 'A', map_irrigation(e, base_rest_data[e], 'A', 'A')) for e in range(1, 13)}
# todo need revisit with any new classifications updated 24-02-2021
base_events[6] = ('C', 'A', 0)
base_events[7] = ('C', 'A', 0)

default_storyline_time = pd.date_range('2024-07-01', '2027-06-01', freq='MS')


def make_irr_rest_for_all_events():
    data = get_past_event_frequency().reset_index()
    rest_keys = [50, 60, 75, 80, 90, 95, 99]
    for i, m, s in data.loc[:, ['month', 'state']].itertuples(True, None):
        for k in rest_keys:
            data.loc[i, k] = map_irrigation(m, k/100, s.split('-')[1].replace('P', ''), 'A')
    out_keys = ['month', 'state'] + rest_keys
    data = data.loc[:, out_keys]
    data.to_csv(os.path.join(climate_shocks_env.supporting_data_dir, 'irrigation_rest_s_a.csv'))
    pass


def make_blank_storyline_sheet():
    opts = make_sampling_options()
    for m in range(1, 13):
        temp = pd.Series(opts[m][:, 0]).str.replace('A', 'AT')
        precip = pd.Series(opts[m][:, 1]).str.replace('A', 'AP')
        opts[m] = '\n'.join(['-'.join([t, p]) for t, p in zip(temp, precip)])

    outdata = pd.DataFrame(index=default_storyline_time)
    outdata.loc[:, 'year'] = outdata.index.year
    outdata.loc[:, 'month'] = outdata.index.month_name().str[:3]
    months = outdata.index.month
    outdata.loc[:, 'options'] = [opts[e] for e in months]
    return outdata


if __name__ == '__main__':
    make_irr_rest_for_all_events()
