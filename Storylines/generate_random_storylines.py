"""
 Author: Matt Hanson
 Created: 12/01/2021 10:39 AM
 """

import pandas as pd
import numpy as np
import os
from copy import deepcopy
from Storylines.check_storyline import get_past_event_frequency
from Storylines.storyline_building_support import map_storyline_rest
from Climate_Shocks.climate_shocks_env import temp_storyline_dir


# make each storyline based on random weather, but tied to the probability of events, e.g. average is more likely  than not.
# get rid of any duplicates!
# make each storyline a pick of a bunch of restriction profiles (hand made to allow autocorrelation
# make each storyline 1 year and then make random 3 year combinations of these.
# how do I want to store this? make 12 random seeds?, how will this affect the randomness...



def generate_random_weather(n, use_default_seed=True):
    """
    generate n psuedo-random 1 year weather series. in the storyline format.  the weather is sampled with the
    de-trended historical probailities for each temperature and precip state.
    :param n: number of series to generate
    :param use_default_seed: boolean if True use the saved seeds to generate the distribution, which allows
                             reproducability
    :return: array of T-P states shape (n, 12)
    """
    assert isinstance(n, int)
    # get seeds
    if use_default_seed:
        seeds = np.array([158801, 63516, 107565, 467150, 458444, 388162, 232880, 261663,
                          165845, 228439, 216702, 329875])
    else:
        seeds = np.random.randint(1, 500000, (12,))

    # get states and probabilities
    past_freq = (get_past_event_frequency() / 48).reset_index().set_index('month')
    # this is from the detreneded2 data as that was used to make event def
    states, probs = [], []
    for m in range(1, 13):
        states.append([e.replace('P', '').replace('T', '') for e in past_freq.loc[m, 'state'].values])
        probs.append(past_freq.loc[m, 'year'].values)  # year as it is just sloppy calc of frequency

        out = []
        # generate the suites
        for s, m, state, prob in zip(seeds, range(1, 13), states, probs):
            np.random.seed(s)
            out.append(np.random.choice(state, size=(n, 1), p=prob))
        out = np.concatenate(out, axis=1)
    return out


def generate_irrigation_suites():  # todo this is the challenge! work with 10 percentile jumps from median..., purely random combinations.
    # shape = (n,12)
    raise NotImplementedError


def generate_random_suite(n, use_default_seed=True, save=True, return_story=False):
    if save:
        outdir = os.path.join(temp_storyline_dir, 'random')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    if use_default_seed:
        wseed = 106580
        irseed = 310088

    else:
        wseed = np.random.randint(1, 500000)
        irseed = np.random.randint(1, 500000)

    irrigation = generate_irrigation_suites()
    irr_len = len(irrigation)
    weather = generate_random_weather(n, use_default_seed)
    wea_len = len(weather)  # should be n, but for code clarity

    # generate random options
    np.random.seed(wseed)
    out_weathers = np.random.randint(0, wea_len, size=(n, 3))
    np.random.seed(irseed)
    out_irr = np.random.randint(0, irr_len, size=(n, 3))

    out_idxs = np.concatenate([out_weathers, out_irr], axis=1)

    # get rid of duplicates
    out_idxs = np.unique(out_idxs, axis=0)
    assert out_idxs.shape[1] == 6

    data = pd.DataFrame(index=pd.date_range('2024-07-01', '2027-06-01', freq='MS'),
                        columns=['precip_class', 'temp_class', 'rest'])
    data.index.name = 'date'
    data.loc[:, 'year'] = data.index.year
    data.loc[:, 'month'] = data.index.month

    # make into dataframes
    out = []
    for i, (w1, w2, w3, i1, i2, i3) in enumerate(out_idxs):
        # year 1
        temp = deepcopy(data)
        t = pd.Series(weather[w1]).str.split('-').str[0]
        p = pd.Series(weather[w1]).str.split('-').str[1]
        temp.iloc[0:12, 0] = p
        temp.iloc[0:12, 1] = t
        temp.iloc[0:12, 2] = irrigation[i1]

        # year 2
        t = pd.Series(weather[w2]).str.split('-').str[0]
        p = pd.Series(weather[w2]).str.split('-').str[1]
        temp.iloc[12:24, 0] = p
        temp.iloc[12:24, 1] = t
        temp.iloc[12:24, 2] = irrigation[i2]

        # year3
        t = pd.Series(weather[w3]).str.split('-').str[0]
        p = pd.Series(weather[w3]).str.split('-').str[1]
        temp.iloc[24:36, 0] = p
        temp.iloc[24:36, 1] = t
        temp.iloc[24:36, 2] = irrigation[i3]

        map_storyline_rest(temp)

        if save:
            temp.to_csv(os.path.join(outdir, f'rs-{i:010d}'))

        if return_story:
            out.append(temp)
    if return_story:
        return out
