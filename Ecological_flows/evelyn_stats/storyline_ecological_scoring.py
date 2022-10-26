"""
created Evelyn_Charlesworth 
on: 27/10/2022
"""
"""A python script that takes in the perturbed storyline data and creates scores etc. accordingly.
Similar as ecological_scoring but just takes in different sets of data"""

import kslcore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
from itertools import groupby
from kslcore import KslEnv
from Climate_Shocks.get_past_record import get_vcsn_record, get_restriction_record
from water_temp_monthly import temp_regr


# getting the temperature data into a csv
def get_temp_dataset():
    """a function that gets the daily water temperature dataset"""
    data = get_vcsn_record(version='trended', site='eyrewell')
    data = data.reset_index()
    for d, t in data.loc[:, 'tmin'].items():
        mean_temp = (t + data.loc[d, 'tmax']) / 2
        data.loc[d, 'mean_daily_air_temp'] = mean_temp
    data['date'] = pd.to_datetime(data['date'])
    data.loc[:, 'water_year'] = [e.year for e in (data.loc[:, 'date'].dt.to_pydatetime() + relativedelta(months=6))]
    x = data.loc[:, 'mean_daily_air_temp'].values.reshape(-1, 1)
    data.loc[:, 'mean_daily_water_temp'] = temp_regr.predict(x)
    data = data.loc[:, ['date', 'water_year', 'mean_daily_air_temp', 'mean_daily_water_temp']]
    data.to_csv(kslcore.KslEnv.shared_gdrive.joinpath(
        'Z2003_SLMACC/eco_modelling/stats_info/temperature_data.csv'))
    return data


if __name__ == '__main__':
    get_temp_dataset()