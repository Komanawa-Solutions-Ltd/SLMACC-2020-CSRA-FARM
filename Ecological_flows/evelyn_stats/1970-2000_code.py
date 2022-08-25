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
    # the date range for this specific file is 1970-2000 (hydrological year, so up to June 30 2000 included)
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

    # Calculating the WUAs for a given flow, for different species
    # Using a nested function

    def flow_to_wua(flow, species_type):
        """ A function that takes in a flow and a species name, and based
        on a polynomial eq, outputs a WUA"""

        x = flow

        # Setting up the correct flow ranges
        if species_type == "bird":
            min_range = 35
            max_range = 85
        else:
            min_range = 18
            max_range = 180

        # Calculating the WUA for each species based on the input flow

        if flow > max_range or flow < min_range:
            print("Sorry, flow is out of range")
        else:
            # These are all for WUA units in m2/m
            longfin_eel = -9.045618237519400E-09 * x ** 5 + 3.658952327544510E-06 * x ** 4 - 5.653574369241410E-04 * x ** 3 + 3.858556802202370E-02 * x ** 2 - 3.239955996233250E-01 * x + 9.987638834796250E+01
            shortfin_eel = -5.964114493071940E-09 * x ** 5 + 2.359764378654360E-06 * x ** 4 - 3.693579872009160E-04 * x ** 3 + 2.683927613703320E-02 * x ** 2 - 3.681012446881110E-01 * x + 8.593725263391190E+01
            torrent_fish = 2.896163694304270E-08 * x ** 5 - 1.167620629575640E-05 * x ** 4 + 1.801041895279500E-03 * x ** 3 - 1.329402534268910E-01 * x ** 2 + 5.277167341236740E+00 * x - 1.408366189647840E+01
            common_bully = 3.679138046845140E-09 * x ** 5 - 1.938607130429040E-07 * x ** 4 - 1.923502238925680E-04 * x ** 3 + 2.961375443166340E-02 * x ** 2 - 1.112066360882710E+00 * x + 7.329526111040610E+01
            upland_bully = -1.670386190380080E-08 * x ** 5 + 7.480690123013630E-06 * x ** 4 - 1.257177384401630E-03 * x ** 3 + 9.648051249735090E-02 * x ** 2 - 3.077836962111130E+00 * x + 8.675954558492810E+01
            bluegill_bully = -6.471586231748120E-09 * x ** 5 + 1.973356622447410E-06 * x ** 4 - 1.949914099179170E-04 * x ** 3 + 5.570337619808730E-03 * x ** 2 + 3.944431105242500E-01 * x + 3.459956435653860E+01
            food_production = 2.130431975429750E-08 * x ** 5 - 9.085807849474580E-06 * x ** 4 + 1.464737145368640E-03 * x ** 3 - 1.125512066047600E-01 * x ** 2 + 4.823875351509410E+00 * x + 1.115625470423880E+01
            brown_trout_adult = 4.716969949537670E-09 * x ** 5 - 2.076496120868080E-06 * x ** 4 + 3.361640291880770E-04 * x ** 3 - 2.557607121249140E-02 * x ** 2 + 1.060052581008110E+00 * x + 3.627596900757210E+00
            chinook_salmon_juv = 6.430228856812380E-09 * x ** 5 - 1.901413063448040E-06 * x ** 4 + 1.779162094752800E-04 * x ** 3 - 5.287064285669480E-03 * x ** 2 + 6.690264788207550E-02 * x + 2.160739430906840E+01
            black_fronted_tern = 1.860374649942380E-06 * x ** 5 - 6.206129788530580E-04 * x ** 4 + 8.139025742665820E-02 * x ** 3 - 5.222181017852630E+00 * x ** 2 + 1.629785813832450E+02 * x - 1.908796384066770E+03

            # This is WUA units in m2*1000
            wrybill_plover = 1.992991145099990E-05 * x ** 5 - 6.562761816460400E-03 * x ** 4 + 8.524578863075030E-01 * x ** 3 - 5.444835223306980E+01 * x ** 2 + 1.702284174702220E+03 * x - 2.058208449588460E+04

            # This is proportion of WUA
            diatoms = 7.415806641571640E-11 * x ** 5 - 3.448627575182280E-08 * x ** 4 + 6.298888857172090E-06 * x ** 3 - 5.672527158325650E-04 * x ** 2 + 2.595917911761800E-02 * x - 1.041530354852930E-01
            long_filamentous = -2.146620894005660E-10 * x ** 5 + 8.915219136657130E-08 * x ** 4 - 1.409667339556760E-05 * x ** 3 + 1.057153790947640E-03 * x ** 2 - 3.874332961128240E-02 * x + 8.884973169426100E-01
            short_filamentous = 1.411793860210670E-10 * x ** 5 - 5.468836816918290E-08 * x ** 4 + 7.736645471349440E-06 * x ** 3 - 4.767919019192250E-04 * x ** 2 + 1.082051321324740E-02 * x + 3.578139911667070E-01


        # Assigning the maximum WUA for each species, based on the original graphs
        # Doing this as a dictionary and then turning into a dataframe
        max_WUA_dict = {"longfin eel": 176, "shortfin eel": 132, "torrent fish": 120, "common bully": 87,
                        "upland bully": 71, "bluegill bully": 75, "food production" : 150, "brown trout adult": 30,
                        "chinook salmon juvenile": 27, "diatoms":0.42, "long filamentous": 0.42, "short filamentous":0.43,
                        "black fronted tern" : 66, "wrybill plover": 203}
        WUA_df = pd.DataFrame(list(max_WUA_dict.items()), columns=['Species', 'Max WUA'])

        # Adding the WUA based on the input flow to the dataframe
        WUA_df['Calculated WUA'] = [longfin_eel, shortfin_eel, torrent_fish, common_bully,
                                    upland_bully,bluegill_bully, food_production, brown_trout_adult,
                                    chinook_salmon_juv, diatoms, long_filamentous, short_filamentous,
                                    black_fronted_tern, wrybill_plover]
        # Calculating the percentage

        WUA_df['Percentage score'] = WUA_df['Calculated WUA'] / WUA_df['Max WUA'] * 100

        # Getting the range - this is just testing
        min_WUA = WUA_df['Percentage score'].min()
        max_WUA = WUA_df['Percentage score'].max()
        range_WUA = max_WUA - min_WUA
        print(max_WUA, min_WUA, range_WUA)


    flow_to_wua(60, "bird")

# NB going to have to iterate through both ALFs AND species names to get this function to work??
read_and_stats('period_a.csv')