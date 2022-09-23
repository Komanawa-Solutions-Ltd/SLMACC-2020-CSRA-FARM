"""
created Evelyn_Charlesworth
on: 25/08/2022
"""
import kslcore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from itertools import groupby
from kslcore import KslEnv


def _wua_poly(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

species_coeffs = {
    "longfin_eel": (-9.045618237519400E-09, 3.658952327544510E-06,
                    5.653574369241410E-04, 3.858556802202370E-02,
                    3.239955996233250E-01, 9.987638834796250E+01),
    "shortfin_eel": (-5.964114493071940E-09, + 2.359764378654360E-06,
                     - 3.693579872009160E-04, + 2.683927613703320E-02,
                     - 3.681012446881110E-01,  + 8.593725263391190E+01),
    "torrent_fish": (2.896163694304270E-08, 1.167620629575640E-05,
                     + 1.801041895279500E-03, - 1.329402534268910E-01,
                     + 5.277167341236740E+00, - 1.408366189647840E+01),
    "common_bully": (3.679138046845140E-09, 1.938607130429040E-07,
                     - 1.923502238925680E-04, + 2.961375443166340E-02,
                     - 1.112066360882710E+00, + 7.329526111040610E+01),
    "upland_bully": (-1.670386190380080E-08, + 7.480690123013630E-06,
                     - 1.257177384401630E-03, + 9.648051249735090E-02,
                     - 3.077836962111130E+00, + 8.675954558492810E+01),
    "bluegill_bully": (-6.471586231748120E-09, + 1.973356622447410E-06,
                       - 1.949914099179170E-04, + 5.570337619808730E-03,
                       + 3.944431105242500E-01, + 3.459956435653860E+01),
    "food_production": (2.130431975429750E-08, - 9.085807849474580E-06,
                        + 1.464737145368640E-03, - 1.125512066047600E-01,
                        + 4.823875351509410E+00, + 1.115625470423880E+01),
    "brown_trout_adult": (4.716969949537670E-09, - 2.076496120868080E-06,
                          + 3.361640291880770E-04, - 2.557607121249140E-02,
                          + 1.060052581008110E+00, + 3.627596900757210E+0),
    "chinook_salmon_junior": (6.430228856812380E-09, - 1.901413063448040E-06,
                              + 1.779162094752800E-04, - 5.287064285669480E-03,
                              + 6.690264788207550E-02, + 2.160739430906840E+01),
    "diatoms": (7.415806641571640E-11, - 3.448627575182280E-08,
                + 6.298888857172090E-06, - 5.672527158325650E-04,
                + 2.595917911761800E-02, - 1.041530354852930E-01),
    "long_filamentous": (-2.146620894005660E-10, + 8.915219136657130E-08,
                         - 1.409667339556760E-05, + 1.057153790947640E-03,
                         - 3.874332961128240E-02, + 8.884973169426100E-01),
    "short_filamentous": (1.411793860210670E-10, - 5.468836816918290E-08,
                          + 7.736645471349440E-06, - 4.767919019192250E-04,
                          + 1.082051321324740E-02, + 3.578139911667070E-01),
    "black_fronted_tern": (1.860374649942380E-06, - 6.206129788530580E-04,
                           + 8.139025742665820E-02, - 5.222181017852630E+00,
                           + 1.629785813832450E+02, - 1.908796384066770E+03),
    "wrybill_plover": (1.992991145099990E-05, - 6.562761816460400E-03,
                       + 8.524578863075030E-01, - 5.444835223306980E+01,
                       + 1.702284174702220E+03, - 2.058208449588460E+04)



}
species_limits = {
    "longfin_eel": (18, 180), "shortfin_eel": (18, 180), "torrent_fish": (18, 180),
    "common_bully": (18, 180), "upland_bully": (18, 180), "bluegill_bully": (18, 180),
    "food_production": (18, 180), "brown_trout_adult": (18, 180), "chinook_salmon_junior": (18, 180),
    "diatoms": (18, 180), "long_filamentous": (18, 180), "short_filamentous": (18, 180),
"black_fronted_tern": (35, 85), "wrybill_plover": (35, 85)
}

species_max_wua = {
    "longfin_eel": 172, "shortfin_eel": 132, "torrent_fish": 120, "common_bully": 87,
    "upland_bully": 71, "bluegill_bully": 75, "food_production": 150, "brown_trout_adult": 30,
"chinook_salmon_junior": 27, "diatoms": 0.42, "long_filamentous": 0.42, "short_filamentous": 0.43,
"black_fronted_tern": 66, "wrybill_plover": 203
}

#todo change so that the max wua is for malf value

def get_dataset():
    base_path = kslcore.KslEnv.shared_gdrive.joinpath(
        'Z2003_SLMACC/eco_modelling/stats_info/66401_Naturalised_flow.csv')
    data = pd.read_csv(base_path)
    data.loc[:, 'Datetime'] = pd.to_datetime(data.loc[:, 'Datetime'], format='%d/%m/%Y')
    data.loc[:, 'water_year'] = [e.year for e in (data.loc[:, 'Datetime'].dt.to_pydatetime() + relativedelta(months=6))]
    data = data.rename(columns={'Datetime': 'date', 'M3PerSecond': 'flow'})
    data = data.loc[:, ['date', 'flow', 'water_year']]
    return data


def get_seven_day_avg(dataframe):
    """ A function that creates the 7-day rolling avg of flow for each year"""

    # Creating an empty dataframe to append the 7-day avg series to
    all_seven_day_avg = pd.DataFrame()
    # Creating a column name that will increase for each hydrological year
    # Need to iterate through the columns and create a 7-day rolling avg for each yr
    for col in dataframe.columns:
        # Turning the dataframe into a series in order to do the rolling avg
        number_series = dataframe.loc[:, col]
        # Creating a series of moving averages in each window
        rolling_avg = number_series.rolling(7).mean()
        all_seven_day_avg[col] = rolling_avg
    return all_seven_day_avg


def flow_to_wua(alf, species):
    minf, maxf = species_limits[species]
    if alf > minf and alf < maxf:
        wua = _wua_poly(alf, *species_coeffs[species])
        wua_per = wua / species_max_wua[species] * 100
    else:
        wua, wua_per = None, None
    return wua, wua_per



def read_and_stats(outpath, start_water_year, end_water_year, flow_limits=None):
    """
    A function that reads in a file of flows (associated w/ dates) and performs stats on them,
    allowing the outputs to be input into other eqs
    :param outpath: where to save the data
    :param start_water_year: integer start year for the data to analyse
    :param end_water_year:  int end year (inclusive)
    :param flow_limits: None or float values for flow limits to calculate
    :return:
    """

    flow_df = get_dataset()
    list_startdates = range(start_water_year, end_water_year + 1)
    flow_df = flow_df.loc[np.in1d(flow_df.water_year, list_startdates)]

    # Calculating stats

    # Calculating the median flow for all years
    # One value for the entire dataset
    median_flow = flow_df['flow'].median()
    print(f"This is the median {median_flow}")

    # Calculating the ALF
    # One ALF per year

    # First, long to wide by hydrological years using a nested function
    all_hydro_years_df = pd.DataFrame(index=range(1, 367), columns=list_startdates)
    for y in list_startdates:
        l = range(1, len(flow_df.loc[flow_df.water_year == y, 'flow']) + 1)
        all_hydro_years_df.loc[l, y] = flow_df.loc[flow_df.water_year == y, 'flow'].values

    # A list of all the years in the dataset, 1970-2000 in this case
    # Creating an empty dataframe to put the hydro years into from the nested function

    # Iterating through the start year dates using the nested function get_hydrological_year
    # Creating a nested function to get a 7-day rolling average in order to get the ALF

    seven_day_avg_df = get_seven_day_avg(all_hydro_years_df)

    # Calculating the ALFs using a nested function
    outdata = pd.DataFrame(index=list_startdates)
    outdata.index.name = 'water_year'
    # calc alf
    outdata.loc[:, 'alf'] = seven_day_avg_df.min()

    # Getting the MALF
    outdata.loc[:, 'malf'] = malf = outdata['alf'].mean()

    # putting the median in outdata
    outdata.loc[:, 'median'] = median_flow


    # Calculating the days per year spent below MALF
    outdata.loc[:, 'days_below_malf'] = (all_hydro_years_df < malf).sum()
    # calculate flow limits
    if flow_limits is not None:
        flow_limits = np.atleast_1d(flow_limits)
        for f in flow_limits:
            f = round(f)
            outdata.loc[:, f'days_below_{f}'] = (all_hydro_years_df < f).sum()

    # consecutive days
    for y in list_startdates:
        t = all_hydro_years_df.loc[:, y]
        t2 = (t < malf)
        outperiods = []
        day = []
        prev_day = False
        period = 0
        for did, current_day in zip(t2.index, t2):
            if prev_day and current_day:
                outperiods.append(period)
                day.append(did)
            else:
                pass
            if prev_day and not current_day:
                period += 1
            prev_day = current_day
        outperiods = np.array(outperiods)
        day = np.array(day)
        # total days that are consecutive (e.g. 2 days = 1 consecutive day, 3 days below restriction = 2)
        outdata.loc[y, 'consec_days'] = len(outperiods)
        outdata.loc[y, 'num_events'] = len(np.unique(outperiods))

    # Finding the ALF anomaly for the worst 1, 2 and 3 yrs
    # The worst ALF year is min of the alf df
    worst_alf = outdata.loc[:, 'alf'].min()
    # Calculating the anomaly of malf - alf for the worst alf year
    outdata.loc[:, 'anomaly_1'] = anomaly_1 = malf - worst_alf

   # getting the worst 2yr consecutive ALf and worst 3yr
    consecutive_alfs = [2, 3]
    for cy in consecutive_alfs:
        outdata.loc[:, f'rolling_alf_{cy}'] = t = outdata.loc[:, 'alf'].rolling(cy).mean()
        outdata.loc[:, f'worst_rolling_{cy}_alf'] = t.min()
        outdata.loc[:, f'malf_worst_{cy}_anom'] = malf - t.min()

    for sp in species_limits:
        for k, v in outdata.loc[:, 'alf'].itertuples(True, None):
            outdata.loc[k, f'{sp}'] = flow_to_wua(v, sp)

    # todo scoring system function
    col_title = 0
    for col in WUA_percen_df.iloc[:, 1:]:
        list_scores = []
        range_per = WUA_percen_df[col].max() - WUA_percen_df[col].min()
        increments = range_per / 5
        min_val = WUA_percen_df[col].min()
        print(increments)
        col_title += 1
        for value in WUA_percen_df[col]:
            if min_val <= value < (min_val + increments):
                score = 1
                list_scores.append(score)
            elif (min_val + increments) < value < (min_val + (increments * 2)):
                score = 2
                list_scores.append(score)
            elif (min_val + (increments * 2)) < value < (min_val + (increments * 3)):
                score = 3
                list_scores.append(score)
            elif (min_val + (increments * 3)) < value < (min_val + (increments * 4)):
                score = 4
                list_scores.append(score)
            elif (min_val + (increments * 4)) < value <= (min_val + (increments * 5)):
                score = 5
                list_scores.append(score)
            else:
                score = "Outside of range"
                list_scores.append(score)
        WUA_percen_df[col_title] = list_scores

    WUA_percen_df.to_csv("V:\\Shared drives\\Z2003_SLMACC\\eco_modelling\\stats_info\\scores_2000.csv")
    outdata.to_csv(outpath)

    # NB ignore the first values because they are for the ALF


if __name__ == '__main__':
    read_and_stats(None, 2000, 2010) # todo path
