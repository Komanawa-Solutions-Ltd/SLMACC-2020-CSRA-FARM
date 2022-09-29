"""
created Evelyn_Charlesworth
on: 25/08/2022
"""
import kslcore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from itertools import groupby
from kslcore import KslEnv


def _wua_poly(x, a, b, c, d, e, f):
    """a function that reads in coefficients and returns a polynomial with the coeffs
    inserted"""
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
                       + 1.702284174702220E+03, - 2.058208449588460E+04)}
species_limits = {
    "longfin_eel": (18, 130), "shortfin_eel": (18, 130), "torrent_fish": (18, 130),
    "common_bully": (18, 130), "upland_bully": (18, 130), "bluegill_bully": (18, 130),
    "food_production": (18, 130), "brown_trout_adult": (18, 130), "chinook_salmon_junior": (18, 130),
    "diatoms": (18, 130), "long_filamentous": (18, 130), "short_filamentous": (18, 130),
"black_fronted_tern": (35, 85), "wrybill_plover": (35, 85)}

species_baseline_malf_wua = {
    "longfin_eel": 228, "shortfin_eel": 97, "torrent_fish": 141, "common_bully": 65,
    "upland_bully": 55, "bluegill_bully": 52, "food_production": 98, "brown_trout_adult": 22,
"chinook_salmon_junior": 23, "diatoms": 0.35, "long_filamentous": 0.33, "short_filamentous": 0.39,
"black_fronted_tern": 66.24, "wrybill_plover": 202}

species_baseline_min_wua = {
    "longfin_eel": 134, "shortfin_eel": 88, "torrent_fish": 61, "common_bully": 61,
    "upland_bully": 53, "bluegill_bully": 45, "food_production": 77, "brown_trout_adult": 18,
"chinook_salmon_junior": 22, "diatoms": 0.26, "long_filamentous": 0.31, "short_filamentous": 0.36,
"black_fronted_tern": 54, "wrybill_plover": 52}

species_baseline_max_wua = {
    "longfin_eel": 426, "shortfin_eel": 107, "torrent_fish": 395, "common_bully": 77,
    "upland_bully": 62, "bluegill_bully": 57, "food_production": 111, "brown_trout_adult": 25,
"chinook_salmon_junior": 25, "diatoms": 0.38, "long_filamentous": 0.41, "short_filamentous": 0.43,
"black_fronted_tern": 66.39, "wrybill_plover": 211}

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
    else:
        wua = None
    return wua

def flow_to_score(min_wua, max_wua, malf_wua, alf_wua):
    """Calculates a score for each alf calculated wua based
    on the baseline period min, max and malf wua for each species
    :param min_wua: the min wua for the species from the baseline period
    :param max_wua: the max wua ' '
    :param malf_wua: the 'average' wua calculated using the baseline period malf
    :param species: the species
    :param alf_wua: the comparison wua"""

    if alf_wua > malf_wua:
        score = (alf_wua - malf_wua)/(max_wua-malf_wua)
    elif alf_wua < malf_wua:
        score = (alf_wua -malf_wua)/(malf_wua - min_wua)
    else:
        score = 0
    # have adjusted the score based on wanting to score from -3 to 3
    return score*3


def days_below_score(min_days,  max_days, mean_days, days_below):
    """"Calculates a score for each hydrological year based on the no.
    of days below malf and/or the flow_limits value"""
    # the score is positive for lesser days
    # and negative for more days (representative of the fact that more days at low flow is bad)

    if days_below > mean_days:
        score1 = (mean_days - days_below)/(max_days - mean_days)
    elif days_below < mean_days:
        score1 = (mean_days - days_below)/(mean_days-min_days)
    else:
        score1 = 0
    # have adjusted the score based on wanting to score from -3 to 3
    return score1*3

def malf_alf_anomaly_score(min_anomaly, max_anomaly, mean_anomaly, anomaly):
    """A function that creates a score based on the malf - alf anomalies
    :param min_anomaly: the minimum anomaly for the baseline period. this gives a +ve score
    :param max_anomaly the maximum anomaly for the baseline period. this gives a -ve score
    :param malf
    :param your calculated anomaly"""

    if anomaly < mean_anomaly:
        score2 = (mean_anomaly - anomaly)/(mean_anomaly - min_anomaly)
    elif anomaly > mean_anomaly:
        score2 = (mean_anomaly - anomaly)/(max_anomaly - mean_anomaly)
    else:
        score2 = 0
    return score2*3

def event_score(event_min, event_max, event_mean, event_count):
    """A function that calculates the score for the no. of events >= than 7, 14, 21 and 28 days"""

    if event_count > event_mean:
    # this is a negative score - worse if count is higher than mean
        score3 = (event_mean - event_count)/(event_max - event_mean)
    elif event_count < event_mean:
    # this is a postive score - better if count is less than mean
        score3 = (event_mean - event_count)/ (event_mean - event_min)
    else:
        score3 = 0
    return score3*3

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


    # consecutive days for malf
    for y in list_startdates:
        t = all_hydro_years_df.loc[:, y]
        t2 = (t <= malf)
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
        outdata.loc[y, 'malf_consec_days'] = len(outperiods)
        outdata.loc[y, 'malf_num_events'] = len(np.unique(outperiods))

        outperiods_df = pd.DataFrame({'malf_events': outperiods, 'malf_length': 1})
        outdata.loc[y, 'malf_event_lengths'] = '-'.join(
            (outperiods_df.groupby('malf_events').count() + 1).values.astype(str)[:, 0])
        outperiods_df = outperiods_df.groupby('malf_events').count()+1
        outperiods_df = np.array(outperiods_df)
        outdata.loc[y,'malf_events_greater_7'] = (outperiods_df >= 7).sum()
        outdata.loc[y,'malf_events_greater_14'] = (outperiods_df >= 14).sum()
        outdata.loc[y, 'malf_events_greater_21'] = (outperiods_df >= 21).sum()
        outdata.loc[y, 'malf_events_greater_28'] = (outperiods_df >= 28).sum()


        # calculate flow limits
    if flow_limits is not None:
        flow_limits = np.atleast_1d(flow_limits)
        for f in flow_limits:
            f = round(f)
            outdata.loc[:, f'days_below_{f}'] = (all_hydro_years_df < f).sum()

        # consecutive days for flow_limits
            for d in list_startdates:
                test = all_hydro_years_df.loc[:, d]
                test2 = (test <= f)
                outperiods1 = []
                day1 = []
                prev_day1 = False
                period1 = 0
                for did1, current_day1 in zip(test2.index, test2):
                    if prev_day1 and current_day1:
                        outperiods1.append(period1)
                        day1.append(did1)
                    else:
                        pass
                    if prev_day1 and not current_day1:
                        period1 += 1
                    prev_day1 = current_day1
                outperiods1 = np.array(outperiods1)
                day1 = np.array(day1)
                # total days that are consecutive (e.g. 2 days = 1 consecutive day, 3 days below restriction = 2)
                outdata.loc[d, 'flow_limits_consec_days'] = len(outperiods1)
                outdata.loc[d, 'flow_limits_num_events'] = len(np.unique(outperiods1))

                outperiods1_df = pd.DataFrame({'flow_events': outperiods1, 'flow_length': 1})
                outdata.loc[d, 'flow_event_lengths'] = '-'.join(
                    (outperiods1_df.groupby('flow_events').count() + 1).values.astype(str)[:, 0])
                outperiods1_df = outperiods1_df.groupby('flow_events').count() + 1
                outperiods1_df = np.array(outperiods1_df)
                outdata.loc[d, 'flow_events_greater_7'] = (outperiods1_df >= 7).sum()
                outdata.loc[d, 'flow_events_greater_14'] = (outperiods1_df >= 14).sum()
                outdata.loc[d, 'flow_events_greater_21'] = (outperiods1_df >= 21).sum()
                outdata.loc[d, 'flow_events_greater_28'] = (outperiods1_df >= 28).sum()




#fixme might not need this code anymore
    # Finding the ALF anomaly for the worst 1, 2 and 3 yrs
    # The worst ALF year is min of the alf df
    worst_alf = outdata.loc[:, 'alf'].min()
    # Calculating the anomaly of malf - alf for the worst alf year
    outdata.loc[:, 'anomaly_1'] = anomaly_1 = median_flow - worst_alf

   # getting the worst 2yr consecutive ALf and worst 3yr
    consecutive_alfs = [2, 3]
    for cy in consecutive_alfs:
        outdata.loc[:, f'rolling_alf_{cy}'] = t = outdata.loc[:, 'alf'].rolling(cy).mean()
        outdata.loc[:, f'worst_rolling_{cy}_alf'] = t.min()
        outdata.loc[:, f'malf_worst_{cy}_anom'] = median_flow - t.min()

    # getting malf - alf for each year
    for i, a in outdata.loc[:, 'alf'].items():
        outdata.loc[i, 'anomalies'] = malf - a

#todo create the scores - using a function

    for sp in species_limits:
        for k, v in outdata.loc[:, 'alf'].items():
            wua = flow_to_wua(v, sp)
            outdata.loc[k, f'{sp}_wua'] = wua

    for species in species_limits:
        min_wua = species_baseline_min_wua[species]
        max_wua = species_baseline_max_wua[species]
        malf_wua = species_baseline_malf_wua[species]
        for idx, v in outdata.loc[:, f'{species}_wua'].items():
            alf_wua = v
            score = flow_to_score(min_wua, max_wua, malf_wua, alf_wua)
            outdata.loc[idx, f'{species}_score' ] = score

    baseline_days_below_malf = {'min': 0, 'max': 121, 'mean': 20}
    baseline_days_below_flow_lims = {'min': 0, 'max': 141, 'mean': 39}

    for idx1, value in outdata.loc[:, 'days_below_malf'].items():
        min_v, max_v, mean_v = baseline_days_below_malf['min'], baseline_days_below_malf['max'], baseline_days_below_malf['mean']
        days_score = days_below_score(min_v, max_v, mean_v, value)
        outdata.loc[idx1, 'days_below_malf_score'] = days_score
#fixme can make the column name chanageble based on flowlimits (e.g using f{})
#fixme but not a priority
    for idx2, value2 in outdata.loc[:, 'days_below_50'].items():
        min_v1, max_v1, mean_v1 = baseline_days_below_flow_lims['min'], baseline_days_below_flow_lims['max'], \
                               baseline_days_below_flow_lims['mean']
        days_score1 = days_below_score(min_v1, max_v1, mean_v1, value2)
        outdata.loc[idx2, 'days_below_flow_lim_score'] = days_score1

    # getting base anomalies score
    baseline_anomalies = {'min': -18.90492, 'max':18.87149, 'mean': 0}
    for idx3, value3 in outdata.loc[:, 'anomalies'].items():
        min_v2, max_v2, mean_v2 = baseline_anomalies['min'], baseline_anomalies['max'], baseline_anomalies['mean']
        anomalies_score = malf_alf_anomaly_score(min_v2, max_v2, mean_v2, value3)
        outdata.loc[idx3, 'anomalies_score'] = anomalies_score

    baseline_event_length = {'min_malf_events_greater_7': 0, 'max_malf_events_greater_7': 4, 'mean_malf_events_greater_7':1,
                                  'min_malf_events_greater_14': 0, 'max_malf_events_greater_14': 2, 'mean_malf_events_greater_14':0.387096774,
                                  'min_malf_events_greater_21': 0, 'max_malf_events_greater_21': 2, 'mean_malf_events_greater_21':0.161290323,
                                  'min_malf_events_greater_28': 0, 'max_malf_events_greater_28': 2, 'mean_malf_events_greater_28':0.161290323,
                             'min_flow_events_greater_7': 0, 'max_flow_events_greater_7': 6, 'mean_flow_events_greater_7':1.6451613,
                                  'min_flow_events_greater_14': 0, 'max_flow_events_greater_14': 4, 'mean_flow_events_greater_14':0.8709677,
                                  'min_flow_events_greater_21': 0, 'max_flow_events_greater_21': 2, 'mean_flow_events_greater_21':0.516129032,
                                  'min_flow_events_greater_28': 0, 'max_flow_events_greater_28': 2, 'mean_flow_events_greater_28':0.258064516}

    col_names = ['malf_events_greater_7', 'malf_events_greater_14', 'malf_events_greater_21', 'malf_events_greater_28',
                 'flow_events_greater_7', 'flow_events_greater_14', 'flow_events_greater_21', 'flow_events_greater_28']

    for c in col_names:
        for idx4, value4 in outdata.loc[:, c].items():
            min_v3 = baseline_event_length[f'min_{c}']
            max_v3 = baseline_event_length[f'max_{c}']
            mean_v3 = baseline_event_length[f'mean_{c}']
            event_length = value4
            event_score_output = event_score(min_v3, max_v3, mean_v3, event_length)
            outdata.loc[idx4, f'{c}_score'] = event_score_output
    #plotting e.gs
    #sns.lineplot(data=outdata[['malf', 'alf']])
    #sns.lineplot(data=outdata[['longfin_eel_wua', 'shortfin_eel_wua', 'torrent_fish_wua', 'common_bully_wua','upland_bully_wua', 'bluegill_bully_wua']])
    #plt.show()


    return outdata
    outdata.to_csv(outpath)




if __name__ == '__main__':
    read_and_stats(kslcore.KslEnv.shared_gdrive.joinpath('Z2003_SLMACC/eco_modelling/stats_info'), 1970, 2000, 50)
