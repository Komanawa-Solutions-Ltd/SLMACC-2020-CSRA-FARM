"""
created Evelyn_Charlesworth 
on: 25/10/2022
"""
"""Getting the maximum daily temp from the Waiau water temp data to then 
compare max daily with mean daily air temp through regression"""

import pandas as pd
from komanawa import kslcore
from sklearn.linear_model import LinearRegression

#base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_Uwha_tidied.csv")
#data = pd.read_csv(base_path)
# converting into PeriodIndex and then getting the daily max

# commented out so don't have to keep rerunning, just reading in instead
#daily_max_df = data.groupby(pd.PeriodIndex(data["Date & Time"], freq="D"))['Water Temp (degC)'].max()
#save_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_max.csv")
#daily_max_df.to_csv(save_path)

# reading in the daily air temp

water_base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_max.csv")
daily_max_water_df = pd.read_csv(water_base_path)
base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/waiau_temp/daily_temperature_Cheviot Ews_data.csv")
mean_air_temp_df = pd.read_csv(base_path)
mean_air_temp_df = mean_air_temp_df.rename(columns={'time': 'date'})
mean_air_temp_df['date'] = pd.to_datetime(mean_air_temp_df['date'], format='%Y/%m/%d')

#

daily_max_water_df['Date & Time'] = pd.to_datetime(daily_max_water_df['Date & Time'], format='%Y/%m/%d')
daily_max_water_df = daily_max_water_df.rename(columns={'Date & Time' : 'date', 'Water Temp (degC)': 'daily_max_water_temp'})
merged_df = daily_max_water_df.merge(mean_air_temp_df)

regression_df = merged_df.dropna()
x = regression_df.loc[:,'daily_mean_air_temp'].values.reshape(-1, 1)
y = regression_df.loc[:,'daily_max_water_temp'].values.reshape(-1, 1)
temp_regr = LinearRegression()
temp_regr.fit(x, y)
#print(temp_regr.score(x,y))


daily_mean_water_df = pd.read_csv(kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_mean.csv"))

pass