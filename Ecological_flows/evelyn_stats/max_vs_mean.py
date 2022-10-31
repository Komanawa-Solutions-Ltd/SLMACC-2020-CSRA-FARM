"""
created Evelyn_Charlesworth 
on: 1/11/2022
"""
# looking at the relationship between daily max and daily mean temperature for the waiau
# to see if there is a relationship which can then be applied to the waimakariri

import pandas as pd
import numpy as np
import kslcore
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# reading in the predicted Waimakariri mean daily water temp
# uses the eyrewell daily mean air temp data to predict daily mean water temp data based on the temperature regression created in
# water_temp_monthly.py

waimak_daily_mean_water = pd.read_csv(kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/waimak_daily_mean_temp.csv"))
# getting a month column
waimak_daily_mean_water.loc[:, 'month'] = pd.DatetimeIndex(waimak_daily_mean_water['date']).month

# reading in the daily mean water df
daily_mean_water_df = pd.read_csv(kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_mean.csv"))
daily_mean_water_df = daily_mean_water_df.rename(columns={'Water Temp (degC)': 'daily_mean_water_temp'})

# reading in the daily max water df
daily_max_water_df = pd.read_csv(kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_max.csv"))
daily_max_water_df = daily_max_water_df.rename(columns={'Water Temp (degC)': 'daily_max_water_temp'})

# merging the two to get the same date range
daily_mean_and_max_temp = daily_max_water_df.merge(daily_mean_water_df, on='Date & Time')

# now getting a relationship per month
daily_mean_and_max_temp.loc[:, 'month'] = pd.DatetimeIndex(daily_mean_and_max_temp['Date & Time']).month
# subsetting for each month

# getting the regression between the max and mean
# and predicting the waimakariri data? attempting


month_list = {'jan': 1, 'feb': 2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

predicted_waimak_df = pd.DataFrame()

for month, num in month_list.items():
    # subset of the waiau data that we want to use to fit
    waiau_monthly_df = daily_mean_and_max_temp[daily_mean_and_max_temp['month'] == num]
    waiau_monthly_df = waiau_monthly_df.dropna()
    # subset of the waikmakariri data that we want to use to predict
    waimak_monthly_temp = waimak_daily_mean_water[waimak_daily_mean_water['month'] == num]
    waimak_monthly_temp = waimak_monthly_temp.dropna()
    # reshaping for regression
    x = waiau_monthly_df.loc[:, 'daily_mean_water_temp'].values.reshape(-1, 1)
    y = waiau_monthly_df.loc[:, 'daily_max_water_temp'].values.reshape(-1, 1)
    z = waimak_monthly_temp.loc[:, 'mean_daily_water_temp'].values.reshape(-1, 1)
    max_mean_temp_regr = LinearRegression()
    # fitting the regression between the waiau daily mean water temp and daily max water temp
    max_mean_temp_regr.fit(x, y)
    # using the relationship to predict the waimakariri max daily water temp from the mean daily water temp
    waimak_monthly_temp.loc[:, 'predicted_daily_max_water_temp'] = max_mean_temp_regr.predict(z)
    frames = [predicted_waimak_df, waimak_monthly_temp]
    # concatenating the dataframes together so that at the end we have all the months
    # each month is predicted by its individual relationship
    predicted_waimak_df = pd.concat(frames)
    print(max_mean_temp_regr.score(x,y))
    #plt.scatter(x, y)
    #plt.show()

predicted_waimak_df.to_csv(kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/waimak_daily_max_temp_predicted.csv"))



pass
