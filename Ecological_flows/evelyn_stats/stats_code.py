"""
created Evelyn_Charlesworth 
on: 19/08/2022
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# first create a function that allows the data to be read in and stats to be performed
# if necessary, re-code as separate or nested functions later on

def read_and_stats(file, pathway="V:\\Shared drives\\Z2003_SLMACC\\eco_modelling\\stats_info\\"):
    """A function that reads in a file of flows (associated w/ dates) and performs stats on them,
    allowing the outputs to be input into other eqs"""

    # Default pathway is to the project folder
    # Make sure to double backslash (\\) any pathways

    flow_df = pd.read_csv(pathway + file, parse_dates=['Date'], index_col='Date')
    # This dataframe has a date and a flow column

    # Calculating stats

    # Calculating the median flow for all years
    # One value for the entire dataset
    median_flow = flow_df['Flow'].median()
    # Printing a string that
    median_flow_output = f"The median flow across all years is {median_flow:.2f} m3/s"
    print(median_flow_output)

    # Calculating the average low flow (ALF) for each year
    # The no. of ALFs = no. of years in data set
    #todo need to figure out how to start in break it up into years as well
    # This could be a nested function
    # Automate this process more

    # Figure out how to potentially put these all into one?
    # Or if the process was automated/in a function might not need to
    year_1 = flow_df.loc["1972-07-15":"1973-07-15"]
    year_2 = flow_df.loc["1973-07-15":"1974-07-15"]
    year_3 = flow_df.loc["1974-07-15":"1975-07-15"]
    year_4 = flow_df.loc["1975-07-15":"1976-07-15"]
    year_5 = flow_df.loc["1976-07-15":"1977-07-15"]
    year_6 = flow_df.loc["1977-07-15":"1978-07-15"]
    year_7 = flow_df.loc["1978-07-15":"1979-07-15"]
    year_8 = flow_df.loc["1979-07-15":"1980-07-15"]
    year_9 = flow_df.loc["1980-07-15":"1981-07-15"]
    year_10 = flow_df.loc["1981-07-15":"1982-07-15"]
    year_11 = flow_df.loc["1982-07-15":"1983-07-15"]
    year_12 = flow_df.loc["1983-07-15":"1984-07-15"]
    year_13 = flow_df.loc["1984-07-15":"1985-07-15"]
    year_14 = flow_df.loc["1985-07-15":"1986-07-15"]
    year_15 = flow_df.loc["1986-07-15":"1987-07-15"]
    year_16 = flow_df.loc["1987-07-15":"1988-07-15"]
    year_17 = flow_df.loc["1988-07-15":"1989-07-15"]
    year_18 = flow_df.loc["1989-07-15":"1990-07-15"]
    year_19 = flow_df.loc["1990-07-15":"1991-07-15"]
    year_20 = flow_df.loc["1991-07-15":"1992-07-15"]
    year_21 = flow_df.loc["1992-07-15":"1993-07-15"]
    year_22 = flow_df.loc["1993-07-15":"1994-07-15"]
    year_23 = flow_df.loc["1994-07-15":"1995-07-15"]
    year_24 = flow_df.loc["1995-07-15":"1996-07-15"]
    year_25 = flow_df.loc["1996-07-15":"1997-07-15"]
    year_26 = flow_df.loc["1997-07-15":"1998-07-15"]
    year_27 = flow_df.loc["1998-07-15":"1999-07-15"]
    year_28 = flow_df.loc["1999-07-15":"2000-07-15"]
    year_29 = flow_df.loc["2000-07-15":"2001-07-15"]
    year_30 = flow_df.loc["2001-07-15":"2002-07-15"]
    year_31 = flow_df.loc["2002-07-15":"2003-07-15"]
    year_32 = flow_df.loc["2003-07-15":"2004-07-15"]
    year_33 = flow_df.loc["2004-07-15":"2005-07-15"]
    year_34 = flow_df.loc["2005-07-15":"2006-07-15"]
    year_35 = flow_df.loc["2006-07-15":"2007-07-15"]
    year_36 = flow_df.loc["2007-07-15":"2008-07-15"]
    year_37 = flow_df.loc["2008-07-15":"2009-07-15"]
    year_38 = flow_df.loc["2009-07-15":"2010-07-15"]
    year_39 = flow_df.loc["2010-07-15":"2011-07-15"]
    year_40 = flow_df.loc["2011-07-15":"2012-07-15"]
    year_41 = flow_df.loc["2012-07-15":"2013-07-15"]
    year_42 = flow_df.loc["2013-07-15":"2014-07-15"]
    year_43 = flow_df.loc["2014-07-15":"2015-07-15"]
    year_44 = flow_df.loc["2015-07-15":"2016-07-15"]
    year_45 = flow_df.loc["2016-07-15":"2017-07-15"]
    year_46 = flow_df.loc["2017-07-15":"2018-07-15"]
    year_47 = flow_df.loc["2018-07-15":"2019-07-15"]

    print(type(year_47))
    # Creating a nested function to get a 7-day rolling average
    # and then create ALF

    def 7day_avg(dataframe):
    """A nested function that calculates the 7-day rolling averages for each
    hydrological year """


read_and_stats('initial_flow_data.csv')


