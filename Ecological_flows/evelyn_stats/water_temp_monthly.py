"""
created Evelyn_Charlesworth 
on: 30/08/2022
"""
# Turning 15min water temp data into monthly averages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import groupby
import kslcore
from kslcore import KslEnv

base_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_Uwha_tidied.csv")
data = pd.read_csv(base_path)
# converting into PeriodIndex and then getting the mean

monthly_mean_df = data.groupby(pd.PeriodIndex(data["Date & Time"], freq="M"))['Water Temp (degC)'].mean()

#monthly_mean_df.to_csv("V:\\Shared drives\\Z2003_SLMACC\\eco_modelling\\temp_data\\Waiau_monthly_mean.csv")

daily_mean_df = data.groupby(pd.PeriodIndex(data["Date & Time"], freq="D"))['Water Temp (degC)'].mean()
save_path = kslcore.KslEnv.shared_gdrive.joinpath("Z2003_SLMACC/eco_modelling/temp_data/Waiau_daily_mean.csv")
daily_mean_df.to_csv(save_path)