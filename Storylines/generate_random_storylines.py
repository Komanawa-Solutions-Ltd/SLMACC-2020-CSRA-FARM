"""
 Author: Matt Hanson
 Created: 12/01/2021 10:39 AM
 """

import pandas as pd
import numpy as np
import os
from copy import deepcopy

import ksl_env
from Storylines.check_storyline import get_past_event_frequency
from Storylines.storyline_building_support import map_storyline_rest, prev_month, month_fchange
from Climate_Shocks.climate_shocks_env import temp_storyline_dir


# make each storyline based on random weather, but tied to the probability of events, e.g. average is more likely  than not.
# get rid of any duplicates!
# make each storyline a pick of a bunch of restriction profiles (hand made to allow autocorrelation
# make each storyline 1 year and then make random 3 year combinations of these.
# how do I want to store this? make 12 random seeds?, how will this affect the randomness...

def generate_random_weather_mcmc(n, use_default_seed=True, nmaxiterations=10000,
                                 recalc=False):  # todo check and incorporate!
    """
    generate random weather, where the weather data in the next month is dependent on the transition probabilites and
    the previous month's state.  data is for July-June and July is specified as 'A-A'
    :param n: number of sims
    :param use_default_seed: if True then use the defualt seeds which are entirely reproducable
    :param nmaxiterations: the maximum iterations allowed in the while loop
    :param recalc: if True then recalc
    :return:
    """
    assert isinstance(n, int)
    save_path = os.path.join(ksl_env.slmmac_dir, 'random_weather', f'random_weather_size_{n}.npy')

    if os.path.exists(save_path) and not recalc:
        outdata = np.load(save_path)
        return outdata

    # get trans_probabilities
    trans_probs = {
        # m, dataframe, dataframe cols/idxs = {t}-{p}
    }
    mapper = {
        "AT,AP": "A-A",
        "AT,D": "A-D",
        "AT,W": "A-W",
        "C,AP": "C-A",
        "C,D": "C-D",
        "C,W": "C-W",
        "H,AP": "H-A",
        "H,D": "H-D",
        "H,W": "H-W"
    }

    for m in range(13):
        temp = pd.read_csv(os.path.join(ksl_env.proj_root,
                                        f'BS_work/IID/TransitionProbabilities/{month_fchange[m]}_transitions.csv'),
                           comment='#', index_col=0)
        temp.index = [mapper[e] for e in temp.index]
        temp.columns = [mapper[e] for e in temp.columns]

    # get seeds
    if use_default_seed:
        np.random.seed(458444)
    seeds = np.random.randint(1, 500000, (n * 12 * 100))
    outdata = np.full(size=(n, 12), fill_value='z-z')
    seed_idx = 0
    for i in range(n):
        np.random.seed(seeds[seed_idx])
        seed_idx += 1
        new_array = np.full(size=(12,), fill_value='x-x')
        new_array[0] = 'A-A'
        breaks_idx = 0
        while (outdata == new_array).all(axis=1).any():  # to prevent duplicates
            breaks_idx += 1
            if breaks_idx > nmaxiterations:
                raise ValueError(f'broken while loop more than {nmaxiterations} iterations')

            for mi, m in enumerate([8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]):
                prev_st_month = prev_month[m]
                prev_state = new_array[mi]
                np.random.seed(seeds[seed_idx])
                seed_idx += 1
                options = trans_probs[prev_st_month].index.values
                probs = trans_probs[prev_st_month][prev_state].values
                new_array[mi + 1] = np.random.choice(options, p=probs)
        outdata[i] = new_array
    np.save(save_path, outdata)
    return outdata


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


def generate_irrigation_suites(n, use_default_seed=True, bad_irr=True):
    """
    Note this will not interlace good and bad irriagion systems, e.g. you can't go from 20th percentile to a 60th
    percentile restrictions. I belive that this is generally reasonable as irrigation restrictions are highly
    autocorrelated, but it is a limitation.
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
                        columns=['precip_class', 'temp_class', 'rest', 'rest_per'])
    data.index.name = 'date'
    data.loc[:, 'year'] = data.index.year - 1  # set to the start of the simulation to 2024 in order to match PGRA
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
        temp.iloc[0:12, 3] = irrigation[i1]

        map_storyline_rest(temp)

        if save:
            temp.to_csv(os.path.join(outdir, f'rs-{i:010d}'))

        if return_story:
            out.append(temp)
    if return_story:
        return out


if __name__ == '__main__':
    out = generate_random_suite(5, save=False, return_story=True, bad_irr=False)
    print(out)
