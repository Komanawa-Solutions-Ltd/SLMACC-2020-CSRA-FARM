"""
created Evelyn_Charlesworth 
on: 25/08/2022
"""

# taking what was done in the original code and optimising it for the specific time period

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import groupby

# Reading in the files

def read_and_stats(file, pathway="V:\\Shared drives\\Z2003_SLMACC\\eco_modelling\\stats_info\\"):
    """A function that reads in a file of flows (associated w/ dates) and performs stats on them,
    allowing the outputs to be input into other eqs"""

    # This code is dependent on the csv read in. The csv file must have two columns
    # one called 'date', one called 'flow' (in m3/s)
    # the date range for this specific file is 1970-2000
    # Default pathway is to the project folder
    # Make sure to double backslash (\\) any pathways

    flow_df = pd.read_csv(pathway + file, parse_dates=['date'], index_col='date', dayfirst=True)

    # Calculating stats

    # Calculating the median flow for all years
    # One value for the entire dataset
    median_flow = flow_df['flow'].median()

    # Calculating the ALF
    # One ALF per year

    # First, splitting the dataset into hydrological years using a nested function
    def get_hydrological_year(dataframe, startyear):
        """A function that can get the hydrological year, where you input a year and a dataframe"""

        # Getting the hydrological year from the dataset

        # Empty dataframe to put the years into
        storage_df = pd.DataFrame()

        # Formulating the hydrological year period
        endyear = startyear + 1
        hydro_year = dataframe.loc[f"{startyear}-07-01": f"{endyear}-06-30"]
        hydro_year = hydro_year.reset_index()
        # The column title is the START year
        # Appending each hydro year to the storage dataframe
        storage_df[startyear] = hydro_year.iloc[:, 1]
        return storage_df

    # A list of all the years in the dataset, 1970-2000 in this case
    list_startdates = [1970, 1971,
                       1972, 1973, 1974, 1975, 1976, 1977,
                       1978, 1979, 1980, 1981, 1982, 1983,1984, 1985,
                       1986, 1987, 1989, 1990, 1991, 1992,
                       1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000]

    # Creating an empty dataframe to put the hydro years into from the nested function
    all_hydro_years_df = pd.DataFrame()
    # Iterating through the start year dates using the nested function get_hydrological_year
    for year in list_startdates:
        x = get_hydrological_year(flow_df, year)
        all_hydro_years_df[year] = x

    # Creating a nested function to get a 7-day rolling average in order to get the ALF

    def get_seven_day_avg(dataframe):
        """ A function that creates the 7-day rolling avg of flow for each year"""

        # Creating an empty dataframe to append the 7-day avg series to
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
        return all_seven_day_avg

    seven_day_avg_df = get_seven_day_avg(all_hydro_years_df)

    # Calculating the ALFs using a nested function
    def get_alf(dataframe):
        """This function calculates the ALF for each hydrological year, which is the minimum value of each column
        e.g the minimum 7day avg per year"""

        # Creating an empty list to store all the ALFs
        alf_list = []
        for col in dataframe:
            alf = dataframe[col].min()
            alf_list.append(alf)
        # turning the list into a df
        # where startdate is the starting year of the hydrological year
        # e.g 1972 is the hydrological year 01-07-1972 to 30-06-1973
        alf_df = pd.DataFrame(alf_list, list_startdates, columns=['ALF'])
        return alf_df

    alf = get_alf(seven_day_avg_df)

    # Getting the MALF
    malf = alf['ALF'].mean()

    # Calculating the days per year spent below MALF
    # Difficult to do as a dataframe, so doing as a list and then turning into a dataframe
    days_per_year_below_malf = []
    for col in all_hydro_years_df:
        days_count = all_hydro_years_df[col] < malf
        total_days = days_count.sum()
        days_per_year_below_malf.append(total_days)
    days_per_year_below_malf_df = pd.DataFrame(days_per_year_below_malf, list_startdates,
                                               columns=['Days below MALF per Year'])

    # Getting the low flow stress days - days below x per year
    # Same process as above
    # Difficult to do as a dataframe, so doing as a list and then turning into a dataframe
    # Starting as x = 50
    x = 50
    days_per_year_stress = []
    for col in all_hydro_years_df:
        days_count_2 = all_hydro_years_df[col] < x
        total_days_2 = days_count_2.sum()
        days_per_year_stress.append(total_days_2)
    days_per_year_stress_df = pd.DataFrame(days_per_year_stress, list_startdates,
                                           columns=['Low Flow Stress Accrual Days'])

    # Finding the ALF anomaly for the worst 1, 2 and 3 yrs
    # The worst ALF year is min of the alf df
    worst_alf = alf['ALF'].min()
    # Calculating the anomaly of malf - alf for the worst alf year
    anomaly_1 = malf - worst_alf

    # Calculating the worst 2 & 3 consecutive years
    # Using a nested function that uses the rolling method

    def get_worst_years(dataframe, no_years):
        """ A function that sums the ALF for each year, depending on the period
        specified - e.g if no_years = 2, flow is summed at 2 year intervals"""

        # Creating an empty dataframe to append the summed flows to
        all_summed = pd.DataFrame()
        # Creating a column name that will increase for each hydrological year
        col_name_1 = 0
        # Need to iterate through the columns and sum for years specified
        for col in dataframe:
            col_name_1 += 1
            # Turning the dataframe into a series in order to do the rolling avg
            s = dataframe.loc[:, col]
            # Creating a series of moving averages in each window
            summed_flow = s.rolling(no_years).sum()
            all_summed[col_name_1] = summed_flow
        return all_summed

    # Using the function to get the worst 2yr consecutive ALf and worst 3yr
    worst_2_df = get_worst_years(alf, 2)
    worst_3_df = get_worst_years(alf, 3)

    # Getting the worst 2 & 3 years
    # This is the smallest consecutive flow value across the no. of years
    worst_2 = worst_2_df[1].min()
    worst_3 = worst_3_df[1].min()

    # todo check what is wanted out of these - make absolute value?
    # and what is the value wanted?
    anomaly_2 = malf - worst_2
    anomaly_3 = malf - worst_3




read_and_stats('period_a.csv')
