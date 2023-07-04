"""
created matt_dumont 
on: 4/07/23
"""
import matplotlib.pyplot as plt

from farm_economic_modelling.base_simple_farm_model import BaseSimpleFarmModel
import numpy as np
import pandas as pd

class DummySimpleFarm(BaseSimpleFarmModel):
    states = {  # i value: (nmiking, stock levels)
        1: ('2aday', 'low'),
        2: ('2aday', 'norm'),
        3: ('1aday', 'low'),
        4: ('1aday', 'norm'),
    }

    month_reset = 7  # trigger farm reset on day 1 in July

    def calculate_feed_needed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 5
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_production(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 2
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_next_state(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 1
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_sup_feed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 5
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_running_cost(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 5
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_debt_servicing(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 5
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def reset_state(self, i_month, ):
        out = np.zeros(self.model_shape[1]) + 2
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out


def test_basic_farm_model():
    all_months = list(range(1, 13)) + list(range(1, 13)) + list(range(1, 13))

    farm = DummySimpleFarm(all_months, istate=np.ones(5), pg=all_months, ifeed=np.arange(5)*200,
                           imoney=np.arange(5)*200, sup_feed_cost=0.4, product_price=3.5,
                           monthly_input=True)
    farm.run_model()
    farm.plot_results('state', 'feed', 'money')
    farm.plot_results('state', 'feed', 'money', sims=None, mult_as_lines=False, twin_axs=False)
    farm.plot_results('feed', 'money', sims=None, mult_as_lines=True, twin_axs=True)
    farm.plot_results('feed', 'money', sims=[1,2], mult_as_lines=True, twin_axs=True)

    # todo test different x in plots

    # todo when done test different x in non line plots

    # todo test save to nc

    # todo test save to csv

    # todo test load from nc


    plt.show()
    pass


if __name__ == '__main__':
    test_basic_farm_model()
