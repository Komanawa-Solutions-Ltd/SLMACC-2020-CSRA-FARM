"""
created Evelyn_Charlesworth 
on: 25/10/2022
"""
"""Getting the max air temp data from Cliflo into easier columns to read"""

import pandas as pd
from komanawa import kslcore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/cheviot_max_temp_data.txt")
max_air_temp_data = pd.read_csv(base_path)
max_air_temp_data['Date(NZST)'] = pd.to_datetime(max_air_temp_data['Date(NZST)'], format='%Y%m%d')
max_air_temp_data = max_air_temp_data[['Date(NZST)', 'Tmax(C)']]
save_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/air_temp_daily_max.csv")
max_air_temp_data.to_csv(save_path)

water_base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_max.csv")
daily_max_water_df = pd.read_csv(water_base_path)
daily_max_water_df['Date & Time'] = pd.to_datetime(daily_max_water_df['Date & Time'], format='%Y-%m-%d')
daily_max_water_df = daily_max_water_df.rename(columns={'Date & Time' : 'date', 'Water Temp (degC)': 'daily_max_water_temp'})
max_air_temp_data = max_air_temp_data.rename(columns={'Date(NZST)': 'date', 'Tmax(C)':'daily_max_air_temp'})
max_air_temp_data = max_air_temp_data[max_air_temp_data.daily_max_air_temp != '-']

merged_maxes = daily_max_water_df.merge(max_air_temp_data)
regression_df = merged_maxes.dropna()

x = regression_df.loc[:,'daily_max_air_temp'].values.reshape(-1, 1)
y = regression_df.loc[:,'daily_max_water_temp'].values.reshape(-1, 1)
temp_regr = LinearRegression()
temp_regr.fit(x, y)
print(temp_regr.score(x,y))

plt.scatter(merged_maxes['daily_max_air_temp'], merged_maxes['daily_max_water_temp'], color="black")
#plt.plot(merged_maxes['daily_max_air_temp'], merged_maxes['daily_max_water_temp'], color="blue", linewidth=3)
plt.show()


pass



