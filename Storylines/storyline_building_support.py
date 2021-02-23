"""
 Author: Matt Hanson
 Created: 19/02/2021 3:38 PM
 """
import warnings
import pandas as pd
import numpy as np
from Storylines.check_storyline import get_acceptable_events
from Climate_Shocks import climate_shocks_env

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

base_events = {e: ('A', 'A', 0) for e in range(1, 13)}
base_events[7] = ('C', 'A', 0)

# make all possible events
def make_sampling_options(include_irr=True):
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
                if m in irrig_season and include_irr:
                    temp.append(k.split('-')[0])
                    precip.append(k.split('-')[1])
                    rest.append(1)

        outdata[m] = np.array([temp, precip, rest]).transpose()

    return outdata

def map_irrigation(m, rest):
    # todo also include the precip state for the month! and previous month???
    # todo map irrigation for the month for True and False states.
    warnings.warn('###########MAPPING RESTRICTIONS IS ONLY AT DUMMY FUNCTINO YOU DUMMY##############')
    return 0

def get_rest_base_data(): #todo should this be defined by the precip and previous precip??? or just the precip??
    data = pd.read_csv(climate_shocks_env.event_def_path, skiprows=1)
    data.loc[:, 'mlen'] = data.loc[:, 'month'].replace(month_len)
    data.loc[:, 'rest'] = data.loc[:, 'rest_cum'] / data.loc[:, 'mlen']
    data = data.loc[(data.temp == 0) & (data.precip == 0)]
    out = data.loc[:, ['month', 'rest']].groupby('month').describe().round(4)

    return out[('rest', 'mean')].to_dict()