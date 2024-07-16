"""
created Evelyn_Charlesworth 
on: 30/08/2022
"""
# Turning 15min water temp data into monthly averages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import groupby
from komanawa import kslcore
from komanawa.kslcore import KslEnv
from sklearn.linear_model import LinearRegression


base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_Uwha_tidied.csv")
data = pd.read_csv(base_path)
## converting into PeriodIndex and then getting the mean
#
#monthly_mean_df = data.groupby(pd.PeriodIndex(data["Date & Time"], freq="M"))['Water Temp (degC)'].mean()
#
##monthly_mean_df.to_csv("V:\\Shared drives\\Z2003_SLMACC\\eco_modelling\\temp_data\\Waiau_monthly_mean.csv")
#
#daily_mean_df = data.groupby(pd.PeriodIndex(data["Date & Time"], freq="D"))['Water Temp (degC)'].mean()
#save_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_mean.csv")
#daily_mean_df.to_csv(save_path)

base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/waiau_temp/daily_temperature_Cheviot Ews_data.csv")
air_temp_df = pd.read_csv(base_path, skiprows=[0])

#testing converting to period time and then getting mean
#air_temp_df = air_temp_df.groupby(pd.PeriodIndex(air_temp_df['time'], freq='D'))['temperature'].mean()
#air_temp_df = pd.DataFrame(air_temp_df)
#air_temp_df.columns = ['daily_mean_air_temp']
#save_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/waiau_temp/daily_temperature_Cheviot Ews_data.csv")
air_temp_df = pd.read_csv(base_path)
air_temp_df = air_temp_df.rename(columns={'time': 'date'})
air_temp_df['date'] = pd.to_datetime(air_temp_df['date'], format='%Y/%m/%d')

water_base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_mean.csv")
daily_water_temp = pd.read_csv(water_base_path)
daily_water_temp['Date & Time'] = pd.to_datetime(daily_water_temp['Date & Time'], format='%Y/%m/%d')
daily_water_temp = daily_water_temp.rename(columns={'Date & Time' : 'date', 'Water Temp (degC)': 'daily_mean_water_temp'})
merged_df = daily_water_temp.merge(air_temp_df)
merged_df.to_csv(kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/waiau_temp/air_water_temp_data.csv"))

regression_df = merged_df.dropna()
x = regression_df.loc[:,'daily_mean_air_temp'].values.reshape(-1, 1)
y = regression_df.loc[:,'daily_mean_water_temp'].values.reshape(-1, 1)
temp_regr = LinearRegression()
temp_regr.fit(x, y)
print(temp_regr.score(x,y))

#sns.regplot(x='daily_mean_air_temp', y='daily_mean_water_temp', data=regression_df)
#plt.show()

def regression_predictor():
    regression_df = merged_df.dropna()
    x = regression_df.loc[:, 'daily_mean_air_temp'].values.reshape(-1, 1)
    y = regression_df.loc[:, 'daily_mean_water_temp'].values.reshape(-1, 1)
    temp_regr = LinearRegression()
    temp_regr.fit(x, y)

pass