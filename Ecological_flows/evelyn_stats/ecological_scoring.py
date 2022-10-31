"""
created Evelyn_Charlesworth
on: 25/08/2022
"""
import kslcore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
from itertools import groupby
from kslcore import KslEnv
from Climate_Shocks.get_past_record import get_vcsn_record, get_restriction_record
from water_temp_monthly import temp_regr


malf_full_nat = 41.63893094
malf_baseline_nat = 42.2007397
malf_climate_nat = 41.13377138

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
                     - 3.681012446881110E-01, + 8.593725263391190E+01),
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
    }
species_limits = {
    "longfin_eel": (18, 130), "shortfin_eel": (18, 130), "torrent_fish": (18, 130),
    "common_bully": (18, 130), "upland_bully": (18, 130), "bluegill_bully": (18, 130),
    "food_production": (18, 130), "brown_trout_adult": (18, 130), "chinook_salmon_junior": (18, 130),
    "diatoms": (18, 130), "long_filamentous": (18, 130), "short_filamentous": (18, 130)}

#species_baseline_malf_wua = {
#    "longfin_eel": 228, "shortfin_eel": 97, "torrent_fish": 141, "common_bully": 65,
#    "upland_bully": 55, "bluegill_bully": 52, "food_production": 98, "brown_trout_adult": 22,
#    "chinook_salmon_junior": 23, "diatoms": 0.35, "long_filamentous": 0.33, "short_filamentous": 0.39,
#    "black_fronted_tern": 66.24, "wrybill_plover": 202}

species_baseline_min_wua = {
    "longfin_eel": 146, "shortfin_eel": 89, "torrent_fish": 71, "common_bully": 61,
    "upland_bully": 53, "bluegill_bully": 46, "food_production": 82, "brown_trout_adult": 19,
    "chinook_salmon_junior": 22, "diatoms": 0.28, "long_filamentous": 0.31, "short_filamentous": 0.36}

species_baseline_max_wua = {
    "longfin_eel": 426, "shortfin_eel": 107, "torrent_fish": 395, "common_bully": 77,
    "upland_bully": 62, "bluegill_bully": 57, "food_production": 111, "brown_trout_adult": 25,
    "chinook_salmon_junior": 25.5, "diatoms": 0.38, "long_filamentous": 0.39, "short_filamentous": 0.43}


def get_flow_dataset():
    base_path = kslcore.KslEnv.shared_gdrive.joinpath(
        'Z2003_SLMACC/eco_modelling/stats_info/66401_Naturalised_flow.csv')
    data = pd.read_csv(base_path)
    data.loc[:, 'Datetime'] = pd.to_datetime(data.loc[:, 'Datetime'], format='%d/%m/%Y')
    data.loc[:, 'water_year'] = [e.year for e in (data.loc[:, 'Datetime'].dt.to_pydatetime() + relativedelta(months=6))]
    data = data.rename(columns={'Datetime': 'date', 'M3PerSecond': 'flow'})
    data = data.loc[:, ['date', 'flow', 'water_year']]
    return data


def get_temp_dataset():
    """a function that gets the daily water temperature dataset"""
    data = get_vcsn_record(version='trended', site='eyrewell')
    data = data.reset_index()
    for d, t in data.loc[:, 'tmin'].items():
        mean_temp = (t + data.loc[d, 'tmax']) / 2
        data.loc[d, 'mean_daily_air_temp'] = mean_temp
    data['date'] = pd.to_datetime(data['date'])
    data.loc[:, 'water_year'] = [e.year for e in (data.loc[:, 'date'].dt.to_pydatetime() + relativedelta(months=6))]
    x = data.loc[:, 'mean_daily_air_temp'].values.reshape(-1, 1)
    data.loc[:, 'mean_daily_water_temp'] = temp_regr.predict(x)
    data = data.loc[:, ['date', 'water_year', 'mean_daily_air_temp', 'mean_daily_water_temp']]
    data.to_csv(kslcore.KslEnv.shared_gdrive.joinpath(
        'Z2003_SLMACC/eco_modelling/temp_data/waimak_mean_temp.csv'))
    return data


def get_measured_flow_dataset():
    record = get_restriction_record('trended')
    record = record.reset_index()
    record.loc[:, 'date'] = pd.to_datetime(record.loc[:, 'date'], format='%d/%m/%Y')
    record.loc[:, 'water_year'] = [e.year for e in (record.loc[:, 'date'].dt.to_pydatetime() + relativedelta(months=6))]
    record = record.loc[:, ['date', 'flow', 'water_year']]
    return record


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


def higher_is_better(min_value, max_value, input_value):
    """A function for all the parameters where a larger value means a higher score e.g +++"""

    # keynote NEW scoring system, mult by -1 to switch directions
    # shift to 0-1
    score = (input_value - min_value) / (max_value - min_value)
    score = (score * 2) - 1  # shift score to -1 to 1

    # have adjusted the score based on wanting to score from -3 to 3
    # rounding to the nearest 0.5
    return round((score * 3) * 2.0) / 2.0


def higher_is_worse(min_value, max_value, input_value):
    """"A function where a larger value means a worse score e.g ---- """

    score1 = ((input_value - min_value) / (max_value - min_value))
    score1 = (score1 * 2) - 1 # shift score to -1 to 1
    # have adjusted the score based on wanting to score from -3 to 3
    # rounding
    return round((score1 * -3) * 2.0) / 2.0


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

    # getting flow data
    # keynote change which function is called based on whether getting naturalised or measured flow
    flow_df = get_measured_flow_dataset()

    list_startdates = range(start_water_year, end_water_year + 1)
    flow_df = flow_df.loc[np.in1d(flow_df.water_year, list_startdates)]

    # getting temperature data
    temperature_df = get_temp_dataset()
    # NB temp data starts at 1972 as earliest date
    temperature_df = temperature_df.loc[np.in1d(temperature_df.water_year, list_startdates)]

    # Calculating stats

    # Calculating the median flow for all years
    # One value for the entire dataset
    median_flow = flow_df['flow'].median()

    # First, long to wide by hydrological years
    all_hydro_years_df = pd.DataFrame(index=range(1, 367), columns=list_startdates)
    for y in list_startdates:
        l = range(1, len(flow_df.loc[flow_df.water_year == y, 'flow']) + 1)
        all_hydro_years_df.loc[l, y] = flow_df.loc[flow_df.water_year == y, 'flow'].values

    temperature_wide_df = pd.DataFrame(index=range(1, 367), columns=list_startdates)
    for x in list_startdates:
        length = range(1, len(temperature_df.loc[temperature_df.water_year == x, 'mean_daily_water_temp']) + 1)
        temperature_wide_df.loc[length, x] = temperature_df.loc[
            temperature_df.water_year == x, 'mean_daily_water_temp'].values

    seven_day_avg_df = get_seven_day_avg(all_hydro_years_df)

    # Calculating the ALFs using a nested function
    outdata = pd.DataFrame(index=list_startdates)
    outdata.index.name = 'water_year'
    # calc alf
    outdata.loc[:, 'alf'] = seven_day_avg_df.min()

    # Getting the MALF
    outdata.loc[:, 'malf'] = malf = malf_climate_nat
    outdata.loc[:, 'measured_malf'] = calculated_malf = outdata['alf'].mean()

    # putting the median in outdata
    outdata.loc[:, 'median'] = median_flow

    # Calculating the days per year spent below MALF
    outdata.loc[:, 'days_below_malf'] = (all_hydro_years_df < malf).sum()

    # getting temperature days
    outdata.loc[:, 'temp_days_above_19'] = (temperature_wide_df > 19).sum()
    outdata.loc[:, 'temp_days_above_21'] = (temperature_wide_df > 21).sum()
    outdata.loc[:, 'temp_days_above_24'] = (temperature_wide_df > 24).sum()

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
        outperiods_df = outperiods_df.groupby('malf_events').count() + 1
        outperiods_df = np.array(outperiods_df)
        outdata.loc[y, 'malf_events_greater_7'] = (outperiods_df >= 7).sum()
        outdata.loc[y, 'malf_events_greater_14'] = (outperiods_df >= 14).sum()
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

    # fixme might not need this code anymore
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

    # getting wua for each species for each alf
    for sp in species_limits:
        for k, v in outdata.loc[:, 'alf'].items():
            wua = flow_to_wua(v, sp)
            outdata.loc[k, f'{sp}_wua'] = wua

    # getting the wua score for each species
    for species in species_limits:
        min_wua = species_baseline_min_wua[species]
        max_wua = species_baseline_max_wua[species]
        for idx, v in outdata.loc[:, f'{species}_wua'].items():
            alf_wua = v
            score = higher_is_better(min_wua, max_wua, alf_wua)
            outdata.loc[idx, f'{species}_score'] = score

    baseline_days_below_malf = {'min': 0, 'max': 70}
    baseline_days_below_flow_lims = {'min': 0, 'max': 108}

    # getting the days below malf score
    for idx1, value in outdata.loc[:, 'days_below_malf'].items():
        min_v, max_v = baseline_days_below_malf['min'], baseline_days_below_malf['max']
        days_score = higher_is_worse(min_v, max_v, value)
        outdata.loc[idx1, 'days_below_malf_score'] = days_score


    # getting the days below flow lim score
    for idx2, value2 in outdata.loc[:, 'days_below_50'].items():
        min_v1, max_v1 = baseline_days_below_flow_lims['min'], baseline_days_below_flow_lims['max']
        days_score1 = higher_is_worse(min_v1, max_v1, value2)
        outdata.loc[idx2, 'days_below_flow_lim_score'] = days_score1

    # getting the anomalies score
    baseline_anomalies = {'min': -18.20, 'max': 16.15}
    for idx3, value3 in outdata.loc[:, 'anomalies'].items():
        min_v2, max_v2 = baseline_anomalies['min'], baseline_anomalies['max']
        anomalies_score = higher_is_worse(min_v2, max_v2, value3)
        outdata.loc[idx3, 'anomalies_score'] = anomalies_score

    # getting the event length score
    baseline_event_length = {'min_malf_events_greater_7': 0, 'max_malf_events_greater_7': 4,
                             'min_malf_events_greater_14': 0, 'max_malf_events_greater_14': 2,
                             'min_malf_events_greater_21': 0, 'max_malf_events_greater_21': 1,
                             'min_malf_events_greater_28': 0, 'max_malf_events_greater_28': 1,
                             'min_flow_events_greater_7': 0, 'max_flow_events_greater_7': 6,
                             'min_flow_events_greater_14': 0, 'max_flow_events_greater_14': 4,
                             'min_flow_events_greater_21': 0, 'max_flow_events_greater_21': 2,
                             'min_flow_events_greater_28': 0, 'max_flow_events_greater_28': 1}

    col_names = ['malf_events_greater_7', 'malf_events_greater_14', 'malf_events_greater_21', 'malf_events_greater_28',
                 'flow_events_greater_7', 'flow_events_greater_14', 'flow_events_greater_21', 'flow_events_greater_28']

    for c in col_names:
        for idx4, value4 in outdata.loc[:, c].items():
            min_v3 = baseline_event_length[f'min_{c}']
            max_v3 = baseline_event_length[f'max_{c}']
            event_length = value4
            event_score_output = higher_is_worse(min_v3, max_v3, event_length)
            outdata.loc[idx4, f'{c}_score'] = event_score_output

    # getting days above 19, 21, 24 temperature score
    baseline_temperature_days = {'min_temp_days_above_19': 0, 'max_temp_days_above_19': 22,
                                 'min_temp_days_above_21': 0, 'max_temp_days_above_21': 3,
                                 'min_temp_days_above_24': 0, 'max_temp_days_above_24': 1}
    temp_col_names = ['temp_days_above_19', 'temp_days_above_21', 'temp_days_above_24']

    for col in temp_col_names:
        for idx5, value5 in outdata.loc[:, col].items():
            min_v4 = baseline_temperature_days[f'min_{col}']
            max_v4 = baseline_temperature_days[f'max_{col}']
            daily_temp = value5
            temp_score = higher_is_worse(min_v4, max_v4, daily_temp)
            outdata.loc[idx5, f'{col}_score'] = temp_score



    #outdata.to_csv(outpath)
    return outdata, temperature_df


if __name__ == '__main__':
    read_and_stats(
        kslcore.KslEnv.shared_gdrive.joinpath('Z2003_SLMACC/eco_modelling/stats_info/measured_full_stats.csv'), 1972,
        2019, 50)

