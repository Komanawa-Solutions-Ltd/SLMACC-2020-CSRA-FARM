"""
created matt_dumont 
on: 3/07/23
"""
import numpy as np
import pandas as pd
from farm_economic_modelling.base_simple_farm_model import BaseSimpleFarmModel


class SimpleDairyFarm(BaseSimpleFarmModel):
    states = {  # i value: (nmiking, stock levels)
        1: ('2aday', 'low'),
        2: ('2aday', 'norm'),
        3: ('1aday', 'low'),
        4: ('1aday', 'norm'),
    }

    month_reset = 8  # trigger farm reset on day 1 in August

    def calculate_feed_needed(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'

        # calculate number of cows / ha
        base_cow = 3.5
        state_cow_modifier = {  # i value: (nmiking, stock levels)
            1: 0.95,  # ('2aday', 'low'),
            2: 1.00,  # ('2aday', 'norm'),
            3: 0.95,  # ('1aday', 'low'),
            4: 1.00,  # ('1aday', 'norm'),
        }
        seasonal_cow_modifer = {
            8: 0.90,
            9: 0.95,
            10: 1.00,
            11: 1.00,
            12: 0.95,
            1: 0.95,
            2: 0.90,
            3: 0.85,
            4: 0.80,
            5: 0.80,
            6: 0.00,
            7: 0.00,
        }
        state_cow_modifier = np.array(state_cow_modifier.values())
        ncows = base_cow * seasonal_cow_modifer[month] * state_cow_modifier[current_state - 1]

        # calculate feed/ cows kgDM/cow
        feed_per_cow_state_modifer = {  # i value: (nmiking, stock levels)
            1: 1.,  # ('2aday', 'low'),
            2: 1.,  # ('2aday', 'norm'),
            3: .90,  # ('1aday', 'low'),
            4: .90,  # ('1aday', 'norm'),
        }

        feed_per_cow_monthly = {  # excludes expected typical supplimnations which are included in base costs:
            8: 13.855,
            9: 14.67,
            10: 14.67,
            11: 16.468,
            12: 17.2735,
            1: 16.625,
            2: 16.8,
            3: 15.39,
            4: 14.904,
            5: 14.904,
            6: 0.00,
            7: 0.00,

        }
        feed_per_cow_state_modifer = np.array(feed_per_cow_state_modifer.values())
        feed_per_cow = feed_per_cow_monthly[month] * feed_per_cow_state_modifer[current_state - 1]

        out = ncows * feed_per_cow  # kgDM/ha

        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_production(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class') # todo

    def calculate_next_state(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class') # todo

    def calculate_sup_feed(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class') # todo

    def calculate_running_cost(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class') # todo

    def calculate_debt_servicing(self, month, current_state):
        assert isinstance(month, int), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class') # todo

    def reset_state(self):
        out = np.zeros(self.model_shape[1]) + 2 # reset to state 2 2 a day milking and normal stocking
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out
