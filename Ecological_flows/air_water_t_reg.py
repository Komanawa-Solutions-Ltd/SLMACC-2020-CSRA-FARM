"""
created matt_dumont 
on: 8/9/24
"""
from komanawa.kslcore import KslEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    daily_mean_water_df = pd.read_csv(
        KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'temp_data', 'Waiau_daily_mean.csv'))

    daily_mean_water_df = daily_mean_water_df.rename(columns={'Water Temp (degC)': 'daily_mean_water_temp'})

    # reading in the daily max water df
    daily_max_water_df = pd.read_csv(
        KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'temp_data', 'Waiau_daily_max.csv'))
    daily_max_water_df = daily_max_water_df.rename(columns={'Water Temp (degC)': 'daily_max_water_temp'})

    # merging the two to get the same date range
    daily_mean_and_max_temp = daily_max_water_df.merge(daily_mean_water_df, on='Date & Time')

    # now getting a relationship per month
    daily_mean_and_max_temp.loc[:, 'month'] = pd.DatetimeIndex(daily_mean_and_max_temp['Date & Time']).month
    pass


def get_air_water_reg():
    """
    get the regression model for mean air temperature to maximum daily water temperature
    :return:
    """

    raise NotImplementedError

if __name__ == '__main__':
    get_data()