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

    flow_df = pd.read_csv(pathway + file, parse_dates=['date'], index_col='date', dayfirst=True)
    # This dataframe has a date and a flow column

    # Calculating stats

    # Calculating the median flow for all years
    # One value for the entire dataset
    median_flow = flow_df['flow'].median()
    # Printing a string that
    median_flow_output = f"The median flow across all years is {median_flow:.2f} m3/s"
    print(median_flow_output)

    # Calculating the average low flow (ALF) for each year
    # The no. of ALFs = no. of years in data set

    # Creating a nested function to subset the data into hydrological years
    def get_hydrological_year(dataframe, startyear):
        """Testing out a function that can get the hydrological year"""

        # Want to do what is done below, but then remove the index, and append the column to a dataframe?
        storage = pd.DataFrame()
        endyear = startyear + 1
        hydro_year = dataframe.loc[f"{startyear}-07-01": f"{endyear}-06-30"]
        hydro_year = hydro_year.reset_index()
        # The column title is the START year
        storage[startyear] = hydro_year.iloc[:, 1]
        return storage

    list_startdates = [1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985,
                       1986, 1987, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
                       2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
                       2017, 2018, 2019]

    all_hydro_years = pd.DataFrame()
    for year in list_startdates:
        x = get_hydrological_year(flow_df, year)
        all_hydro_years[year] = x

    # Creating a nested function to get a 7-day rolling average
    # and then create ALF
    def seven_day_avg(dataframe):
        """ A dataframe that creates the 7 day rolling avg of flow for each year"""

        # Creating an empty dataframe to append the 7 day avg series to
        all_seven_day_avg = pd.DataFrame()
        # Creating a column name that will increase for each hydrological year
        col_name = 0
        # Need to iterate through the columns and create a 7-day rolling avg for each yr
        for col in dataframe:
            col_name += 1
            # Turning the dataframe into a series in order to do the rolling avg
            number_series = dataframe.loc[:, col]
            # Creating a series of moving averages in each window
            rolling_avg = number_series.rolling(7).mean()
            all_seven_day_avg[col_name] = rolling_avg
        print(all_seven_day_avg)

    seven_day_avg(all_hydro_years)




read_and_stats('initial_flow_data.csv')


