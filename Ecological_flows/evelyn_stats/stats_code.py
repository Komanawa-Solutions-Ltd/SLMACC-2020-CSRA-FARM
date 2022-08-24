"""
created Evelyn_Charlesworth 
on: 19/08/2022
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import groupby

# first create a function that allows the data to be read in and stats to be performed
# if necessary, re-code as separate or nested functions later on

def read_and_stats(file, pathway="V:\\Shared drives\\Z2003_SLMACC\\eco_modelling\\stats_info\\"):
    """A function that reads in a file of flows (associated w/ dates) and performs stats on them,
    allowing the outputs to be input into other eqs"""

    # This code is dependent on the csv read in. The csv file must have two columns
    # one called 'date', one called 'flow' (in m3/s)
    # the date range is from 1967 - 2022 (01/01/1967 - 09/08/2022)
    # at the moment code must be updated accordingly if this changes

    # Default pathway is to the project folder
    # Make sure to double backslash (\\) any pathways

    flow_df = pd.read_csv(pathway + file, parse_dates=['date'], index_col='date', dayfirst=True)
    # This dataframe has a date and a flow column

    # Calculating stats

    # Calculating the median flow for all years
    # One value for the entire dataset
    median_flow = flow_df['flow'].median()
    # Printing a string that expresses the median
    median_flow_output = f"The median flow across all years is {median_flow:.2f} m3/s"
    print(median_flow_output)

    # Calculating the average low flow (ALF) for each year
    # The no. of ALFs = no. of years in data set

    # Creating a nested function to subset the data into hydrological years
    def get_hydrological_year(dataframe, startyear):
        """Testing out a function that can get the hydrological year, where you input a year and a dataframe"""

        # Getting the hydrological year from the dataset
        # empty dataframe to put the years into
        storage = pd.DataFrame()
        # creating the hydrological year period
        endyear = startyear + 1
        hydro_year = dataframe.loc[f"{startyear}-07-01": f"{endyear}-06-30"]
        hydro_year = hydro_year.reset_index()
        # The column title is the START year
        # appending each hydro year to the storage dataframe
        storage[startyear] = hydro_year.iloc[:, 1]
        return storage

    # a list of all the years in the dataset
    list_startdates = [1967, 1968, 1969, 1970, 1971,
                       1972, 1973, 1974, 1975, 1976, 1977,
                       1978, 1979, 1980, 1981, 1982, 1983,1984, 1985,
                       1986, 1987, 1989, 1990, 1991, 1992,
                       1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
                       2001, 2002, 2003, 2004, 2005, 2006,
                       2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
                       2017, 2018, 2019, 2020, 2021, 2022]

    # creating an empty dataframe to put the hydro years into from the nested function
    all_hydro_years = pd.DataFrame()
    # iterating through the start year dates using the nested function get_hydrological_year
    for year in list_startdates:
        x = get_hydrological_year(flow_df, year)
        all_hydro_years[year] = x

    # Creating a nested function to get a 7-day rolling average
    # and then create ALF
    def get_seven_day_avg(dataframe):
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
        return all_seven_day_avg

    seven_day_avg = get_seven_day_avg(all_hydro_years)

    def get_alf(dataframe):
        """Calculates the ALF for each hydrological year, which is the minimum value of each column
        e.g the minimum 7day avg"""

        # Creating an empty list to store all the alfs
        alf_list = []
        for col in dataframe:
            alf = dataframe[col].min()
            alf_list.append(alf)
        # turning the list into a df
        # where startdate is the starting year of the hydrological year
        # e.g 1972 is the hydrological year 01-07-1972 to 30-06-1973
        alf_df = pd.DataFrame(alf_list, list_startdates, columns=['ALF'])
        return alf_df

    alf = get_alf(seven_day_avg)
    print(f"The ALFs for each year are:{alf}")

    # Getting the MALF
    malf = alf['ALF'].mean()
    print(f"The MALF is: {malf:.2f} m3/s")

    days_per_year_below_malf = []
    # Finding the number of days per year below MALF
    for col in all_hydro_years:
        days_count = all_hydro_years[col] < malf
        count = days_count.sum()
        days_per_year_below_malf.append(count)
    # Difficult to do as a dataframe, so doing as a list and then turning into a dataframe

    days_per_year_below_malf_df = pd.DataFrame(days_per_year_below_malf, list_startdates, columns=['Days below MALF per Year'])

    print(days_per_year_below_malf_df)

    # finding the low flow stress accrual days
    # same process as above
    # starting as x = 50
    x = 50
    days_per_year_stress = []
    for col in all_hydro_years:
        days_count = all_hydro_years[col] < x
        count = days_count.sum()
        days_per_year_stress.append(count)
    # Difficult to do as a dataframe, so doing as a list and then turning into a dataframe

    days_per_year_stress_df = pd.DataFrame(days_per_year_stress, list_startdates, columns=['Low Flow Stress Accrual Days'])
    print(days_per_year_stress_df)

    # finding ALF anomaly for the worst 1, 2 and 3 yrs
    # worst ALF year is min of alf df
    worst_alf = alf['ALF'].min()
    # calculating the anomaly of malf - alf for the worst alf year
    anomaly_1 = malf - worst_alf

    # calculating the worst 2 & 3 consecutive years


    def get_worst_years(dataframe,no_years):
        """ A function that sums the ALF for each year, depending on the period
        specified - e.g if no_years = 2, flow is summed at 2 year intervals"""

        # Creating an empty dataframe to append the 7 day avg series to
        all_summed = pd.DataFrame()
        # Creating a column name that will increase for each hydrological year
        col_name = 0
        # Need to iterate through the columns and sum for years specified
        for col in dataframe:
            col_name += 1
            # Turning the dataframe into a series in order to do the rolling avg
            s = dataframe.loc[:, col]
            # Creating a series of moving averages in each window
            summed_flow = s.rolling(no_years).sum()
            all_summed[col_name] = summed_flow
        return all_summed

    worst_2_df = get_worst_years(alf, 2)
    worst_3_df = get_worst_years(alf, 3)
    # getting the worst 2 & 3 years
    # this is the smallest consecutive flow value across the no. of years

    worst_2 = worst_2_df[1].min()
    worst_3 = worst_3_df[1].min()

    # check what is wanted out of these - make absolute value?

    anomaly_2 = malf - worst_2
    anomaly_3 = malf - worst_3
    print(anomaly_3, anomaly_2)

read_and_stats('naturalised_flow_data.csv')


