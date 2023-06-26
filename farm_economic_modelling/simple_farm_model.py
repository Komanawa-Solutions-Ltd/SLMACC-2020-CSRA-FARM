"""
created matt_dumont 
on: 26/06/23
"""
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

# DEPRECIATED, kill when finished with class farm models.

states = {
    # i value: (nmiking, stock levels)
    1: ('2aday', 'low'),
    2: ('2aday', 'norm'),
    3: ('1aday', 'low'),
    4: ('1aday', 'norm'),
}

feed_demand = {  # todo does this also vary by doy
    # i value: demand kgDM/cow/day
    1: 20,  # todo dummy values
    2: 20,  # todo dummy values
    3: 20,  # todo dummy values
    4: 20,  # todo dummy values
}

cow_per_ha = {
    # i value: cow/ha
    1: 2,  # todo dummy values
    2: 2,  # todo dummy values
    3: 2,  # todo dummy values
    4: 2,  # todo dummy values
}

milk_production = {  # todo does this also vary by doy
    # i value: kgMS/cow/day
    1: 20,  # todo dummy values
    2: 20,  # todo dummy values
    3: 20,  # todo dummy values
    4: 20,  # todo dummy values
}

# todo per stock unit with assumed dairy cow = set stock units.
# todo basic debt model with perterbation
feed_threshold = None  # todo threshold value for feed to take action
feed_target = None  # todo add feed target for year

dry_period = ()  # todo add dry period for cattle
doy_year_reset = None  # todo add doy reset for year


def simple_farm_model(all_months, pg, istate, ifeed, imoney, feed_cost, milk_price):  # todo make monthy
    """

    :param all_months: doy
    :param pg:
    :param istate: initial state number or np.ndarray shape (nsims,)
    :param ifeed: initial feed number or np.ndarray shape (nsims,)
    :param imoney: initial money number or np.ndarray shape (nsims,)
    :param feed_cost: number or np.ndarray shape (time_len,) or (time_len, nsims) # todo set checks
    :param milk_price: number or np.ndarray shape (time_len,) or (time_len, nsims) # todo set checks
    :return:
    """

    assert all([isinstance(e, np.ndarray) for e in
                [all_months, pg, istate, ifeed, imoney]]), 'inputs must be np.ndarray'
    assert set(all_months).issubset(set(range(1, 366))), f'all_doy must be in range 1-365'
    time_len = len(all_months)
    nsims = len(istate)
    assert set(istate).issubset(set(states.keys())), f'unknown istate: {set(istate)} must be one of {states}'

    # setup key model values
    model_shape = (time_len + 1, nsims)
    model_state = np.full(model_shape, -1, dtype=int)
    model_feed = np.full(model_shape, np.nan)
    model_money = np.full(model_shape, np.nan)

    # setup output values
    model_feed_demand = np.full(model_shape, np.nan)
    model_prod = np.full(model_shape, np.nan)
    model_prod_money = np.full(model_shape, np.nan)
    model_feed_imported = np.full(model_shape, np.nan)
    model_feed_cost = np.full(model_shape, np.nan)
    model_running_cost = np.full(model_shape, np.nan)

    # set initial values
    model_state[0, :] = istate
    model_feed[0, :] = ifeed
    model_money[0, :] = imoney
    all_months = np.concatenate([all_months[0], all_months])

    for day in range(1, time_len + 1):
        doy = all_months[day]

        # set start of day values
        current_money = deepcopy(model_money[day - 1, :])
        current_feed = deepcopy(model_feed[day - 1, :])
        current_state = deepcopy(model_state[day - 1, :])
        next_state = deepcopy(model_state[day - 1, :])

        # pasture growth
        current_feed += pg[day, :]

        # feed cattle
        feed_needed = np.array([feed_demand[s] * cow_per_ha[s] for s in current_state])
        # todo variable percentage of feed is imported and included in the standard running costs (reduce demand by feed)
        #   typical suplementary feed farm model dependent, so need farm model int
        current_feed = current_feed - feed_needed
        model_feed_demand[day, :] = feed_needed

        # produce milk
        milk_produced = np.array([milk_production[s] * cow_per_ha[s] for s in current_state])
        model_milk_prod[day, :] = milk_produced

        # sell milk
        milk_money = milk_produced * milk_price[day]
        model_milk_money[day, :] = milk_money
        current_money += milk_money

        # todo evaluate feed
        idx = current_feed < feed_threshold
        if idx.any():
            # todo figure out which options to set
            buy_feed = idx & False  # todo
            to_1aday = idx & False  # todo
            reduce_herd = idx & False  # todo

            # keynote  only do one at a time
            assert isinstance(buy_feed, np.ndarray), 'buy_feed must be np.ndarray'
            assert isinstance(to_1aday, np.ndarray), 'to_1aday must be np.ndarray'
            assert isinstance(reduce_herd, np.ndarray), 'reduce_herd must be np.ndarray'

            if buy_feed.any(): # todo just bring in the daily demand!
                feed_needed = np.full(nsims, 0)
                feed_needed[buy_feed] = feed_target - current_feed[buy_feed]
                feed_cost = feed_needed * feed_cost[day]
                model_feed_inported[day, :] = feed_needed
                model_feed_cost[day, :] = feed_cost
                current_money -= feed_cost
            if to_1aday.any():
                assert not any(to_1aday & np.in1d(current_state, [3, 4])), 'cannot go from 1aday to 1aday'
                tdix = to_1aday & current_state == 1
                next_state[tdix] = 3
                tdix = to_1aday & current_state == 2
                next_state[tdix] = 4
            if reduce_herd.any():
                assert not any(reduce_herd & np.in1d(current_state, [1, 3])), 'cannot go from low to low'
                tdix = reduce_herd & current_state == 2
                next_state[tdix] = 1
                tdix = reduce_herd & current_state == 4
                next_state[tdix] = 3

        # todo add running_cost

        # todo add debt servicing

        # todo new year? change state
        if doy == doy_year_reset: # todo aug 1 always do.
            raise NotImplementedError  # todo add reset options

        # set key values
        model_state[day, :] = next_state
        model_feed[day, :] = current_feed
        model_money[day, :] = current_money

        pass
