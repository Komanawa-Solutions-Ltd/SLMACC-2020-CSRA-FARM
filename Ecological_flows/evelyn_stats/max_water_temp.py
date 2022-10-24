"""
created Evelyn_Charlesworth 
on: 25/10/2022
"""
"""Getting the maximum daily temp from the Waiau water temp data to then 
compare max daily with mean daily air temp through regression"""

import pandas as pd
import kslcore
from sklearn.linear_model import LinearRegression

base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_Uwha_tidied.csv")
data = pd.read_csv(base_path)
# converting into PeriodIndex and then getting the daily max

daily_max_df = data.groupby(pd.PeriodIndex(data["Date & Time"], freq="D"))['Water Temp (degC)'].max()
save_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_max.csv")
daily_max_df.to_csv(save_path)

# reading in the daily air temp

base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/waiau_temp/daily_temperature_Cheviot Ews_data.csv")
mean_air_temp_df = pd.read_csv(base_path, skiprows=[0])
mean_air_temp_df = mean_air_temp_df.rename(columns={'time': 'date'})
mean_air_temp_df['date'] = pd.to_datetime(mean_air_temp_df['date'], format='%Y/%m/%d')