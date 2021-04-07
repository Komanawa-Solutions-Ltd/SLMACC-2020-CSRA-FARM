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
    generate n psuedo-random 1 year weather series STARTING IN JULY. in the storyline format.  the weather is
    sampled with the  de-trended historical probailities for each temperature and precip state.
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
    out = []
    for m, s in zip([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], seeds):
        state = past_freq.loc[m, 'state'].str.replace('P', '').str.replace('T', '').values
        prob = past_freq.loc[m, 'year'].values  # year as it is just sloppy calc of frequency

        # generate the suites
        np.random.seed(s)
        out.append(np.random.choice(state, size=(n, 1), p=prob))
    out = np.concatenate(out, axis=1)
    return out


def generate_irrigation_suites(n, use_default_seed=True, bad_irr=True):  # todo check good irr!
    """

    :param n: number to generate
    :param use_default_seed: if True then use the default seed so it is reproduceable
    :param bad_irr: bool if True then create irrigation from 50-99th percentile if False 1-50th percentile
    :return:
    """
    # shape = (n,12)
    # autocorrelation between percentiles is 0.5 at 1 month and 0.2 at 2 months, so ignorring this is not soo
    # problematic.
    # this is compared to the auto correlation of the data set of 60% at 1 month and 0.3 at 2 months
    if bad_irr:
        options = np.array([50, 60, 70, 80, 90, 95, 99]) / 100.
    else:
        options = np.array([50, 40, 30, 20, 10, 5, 1]) / 100.
    prob = np.array([10, 10, 10, 10, 5, 4, 1]) * 2 / 100
    if use_default_seed:
        if bad_irr:
            seed = 278160
        else:
            seed = 158295
    else:
        seed = np.random.randint(1, 500000)

    out = np.zeros((n, 12))
    np.random.seed(seed)
    out[:, 2:10] = np.random.choice(options, size=(n, 8), p=prob)  # only sample for irrigation months
    return out


def generate_random_suite(n, use_default_seed=True, save=True, return_story=False, bad_irr=True):
    """

    :param n: number to generate
    :param use_default_seed: if True then use the default seed so it is reproduceable
    :param save: bool if true then save to the temp storyline dirs random_{good|bad}_irr
    :param return_story: bool it True retun the storylines
    :param bad_irr: bool if True then create irrigation from 50-99th percentile if False 1-50th percentile
    :return:
    """
    if save:
        if bad_irr:
            outdir = os.path.join(temp_storyline_dir, 'random_bad_irr')
        else:
            outdir = os.path.join(temp_storyline_dir, 'random_good_irr')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    if use_default_seed:
        if bad_irr:
            wseed = 106580
            irseed = 310088
        else:
            wseed = 49102
            irseed = 5215


    else:
        wseed = np.random.randint(1, 500000)
        irseed = np.random.randint(1, 500000)

    irrigation = generate_irrigation_suites(n, use_default_seed, bad_irr=bad_irr)
    irr_len = len(irrigation)
    weather = generate_random_weather(n, use_default_seed)
    wea_len = len(weather)  # should be n, but for code clarity

    # generate random options
    np.random.seed(wseed)
    out_weathers = np.random.randint(0, wea_len, size=(n, 1))
    np.random.seed(irseed)
    out_irr = np.random.randint(0, irr_len, size=(n, 1))

    out_idxs = np.concatenate([out_weathers, out_irr], axis=1)

    # get rid of duplicates
    out_idxs = np.unique(out_idxs, axis=0)
    assert out_idxs.shape[1] == 2

    data = pd.DataFrame(index=pd.date_range('2025-07-01', '2026-06-01', freq='MS'),
                        columns=['precip_class', 'temp_class', 'rest'])
    data.index.name = 'date'
    data.loc[:, 'year'] = data.index.year
    data.loc[:, 'month'] = data.index.month

    # make into dataframes
    out = []
    for i, (w1, i1) in enumerate(out_idxs):
        # year 1
        temp = deepcopy(data)
        t = pd.Series(weather[w1]).str.split('-').str[0]
        p = pd.Series(weather[w1]).str.split('-').str[1]
        temp.iloc[0:12, 0] = p.values
        temp.iloc[0:12, 1] = t.values
        temp.iloc[0:12, 2] = irrigation[i1]

        map_storyline_rest(temp)

        if save:
            temp.to_csv(os.path.join(outdir, f'rs-{i:010d}'))

        if return_story:
            out.append(temp)
    if return_story:
        return out


if __name__ == '__main__':
    out = generate_random_weather(3)
    print(out)
    out = generate_irrigation_suites(3)
    print(out)
    out = generate_random_suite(3, save=False, return_story=True)
    print(out)
