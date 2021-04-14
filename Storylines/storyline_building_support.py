"""
 Author: Matt Hanson
 Created: 19/02/2021 3:38 PM
 """
import pandas as pd
import os
import numpy as np
from Storylines.check_storyline import get_acceptable_events, get_past_event_frequency
from Climate_Shocks import climate_shocks_env
from Storylines.storyline_params import month_fchange, month_len, prev_month, irrig_season
from Storylines.irrigation_mapper import get_irr_by_quantile


# these are used by other  scripts
month_fchange, month_len, prev_month, irrig_season = month_fchange, month_len, prev_month, irrig_season

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

    return _rest_data[key].loc[rest_quantile, m]



_rest_data = get_irr_by_quantile()


base_rest_data = {
    1: 0.5,
    2: 0.5,
    3: 0.5,
    4: 0.5,
    9: 0.5,
    10: 0.5,
    11: 0.5,
    12: 0.5,

    # non-irrigation seasons, The baseline irrigation on non irrigation months must be zero so as not to calculate irr
    # probability for these months.
    5: 0,
    6: 0,
    7: 0,
    8: 0,
}

base_events = {e: ('A', 'A', map_irrigation(e, base_rest_data[e], 'A', 'A'), base_rest_data[e]) for e in range(1, 13)}
# todo need revisit with any new classifications updated 24-02-2021
base_events[6] = ('C', 'A', 0, 0.5)
base_events[7] = ('C', 'A', 0, 0.5)

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
    outdata.to_csv(os.path.join(climate_shocks_env.supporting_data_dir,'blank_storyline.csv'))
    return outdata


if __name__ == '__main__':
    make_irr_rest_for_all_events()
