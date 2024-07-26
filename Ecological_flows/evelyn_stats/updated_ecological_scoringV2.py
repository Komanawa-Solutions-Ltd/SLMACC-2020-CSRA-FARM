"""
updated from updated_ecological_scoring.py so that data is not read in here and it can all be called in one fuction
through finalised_ecological_model.py
created Evelyn_Charlesworth
on: 22/07/2024
"""

import pandas as pd
import numpy as np
from water_temp_monthly import temp_regr
from max_vs_mean import max_mean_temp_regr
from komanawa.kslcore import KslEnv

malf_baseline_nat = 42.2007397
maf_baseline_nat = 991.5673849310346

def _wua_poly(x, a, b, c, d, e, f):
    """a function that reads in coefficients and returns a polynomial with the coeffs
    inserted"""
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f

species_coeffs = {
    "longfin_eel_<300": (-9.045618237519400E-09, 3.658952327544510E-06,
                    5.653574369241410E-04, 3.858556802202370E-02,
                    3.239955996233250E-01, 9.987638834796250E+01),
    "torrent_fish": (2.896163694304270E-08, 1.167620629575640E-05,
                     + 1.801041895279500E-03, - 1.329402534268910E-01,
                     + 5.277167341236740E+00, - 1.408366189647840E+01),
    "brown_trout_adult": (4.716969949537670E-09, - 2.076496120868080E-06,
                          + 3.361640291880770E-04, - 2.557607121249140E-02,
                          + 1.060052581008110E+00, + 3.627596900757210E+0),
    "diatoms": (7.415806641571640E-11, - 3.448627575182280E-08,
                + 6.298888857172090E-06, - 5.672527158325650E-04,
                + 2.595917911761800E-02, - 1.041530354852930E-01),
    "long_filamentous": (-2.146620894005660E-10, + 8.915219136657130E-08,
                         - 1.409667339556760E-05, + 1.057153790947640E-03,
                         - 3.874332961128240E-02, + 8.884973169426100E-01),

    }
species_limits = {
    "longfin_eel_<300": (18, 130), "torrent_fish": (18, 130),
    "brown_trout_adult": (18, 130), "diatoms": (18, 130), "long_filamentous": (18, 130)}

species_baseline_min_wua = {
    "longfin_eel_<300": 146, "torrent_fish": 71, "brown_trout_adult": 19, "diatoms": 0.28,
     "long_filamentous": 0.31}

# max wua = with the median flow
species_baseline_max_wua = {
    "longfin_eel_<300": 426, "torrent_fish": 395, "brown_trout_adult": 25, "diatoms": 0.38,"long_filamentous": 0.39}



def get_seven_day_avg(dataframe):
    """ A function that creates the 7-day rolling avg of flow for each year"""


    # setting the data column as the index
    dataframe = dataframe.set_index('date')

    rolling_avg = dataframe['flow'].rolling(window=7).mean()

    # create the new df with the rolling avg
    rolling_avg_df = pd.DataFrame({
        'date': rolling_avg.index,
        'rolling_avg_flow': rolling_avg.values
    })

    return rolling_avg_df


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


# todo this needs to be updated to have the following variables: air_temp, r_flow, weightings

def generate_scores(air_temp_data, river_flow_data, weightings=None, flow_limits=None):
    """
    A function that reads in a file of flows (associated w/ dates) and performs stats on them,
    allowing the outputs to be input into other eqs
    :param air_temp_data: dataframe, the air temp data with one column datetime and the other daily air temperature in deg C
    :param river_flow_data: dataframe, the river flow data with one column datetime and the other daily river flow in m3/d
    :param weightings: dict, weightings for each ecological flow component
    :param flow_limits: None or float values for flow limits to calculate
    :return:
    """

    #KEYNOTE STATS START HERE
    # Calculating stats

    if weightings is None:
        weightings = {'longfin_eel_<300_score': 1, 'torrent_fish_score': 1, 'brown_trout_adult_score': 1,
                      'diatoms_score': 1,
                      'long_filamentous_score': 1,
                      'days_below_malf_score': 1, 'days_below_flow_lim_score': 1, 'anomalies_score': 1,
                      'malf_events_greater_7_score': 1, 'malf_events_greater_14_score': 1,
                      'malf_events_greater_21_score': 1, 'malf_events_greater_28_score': 1,
                      'flow_events_greater_7_score': 1, 'flow_events_greater_14_score': 1,
                      'flow_events_greater_21_score': 1, 'flow_events_greater_28_score': 1,
                      'temp_days_above_19_score': 1, 'temp_days_above_21_score': 1, 'temp_days_above_24_score': 1,
                      'days_above_maf': 1, 'flood_anomalies_score': 1, 'malf_times_maf_score': 1}
    else:
        raise NotImplementedError

    # todo check this/find a better way of doing?

    x = air_temp_data.loc[:, 'mean_daily_air_temp'].values.reshape(-1, 1)
    air_temp_data.loc[:, 'mean_daily_water_temp'] = temp_regr.predict(x)
    x2 = air_temp_data.loc[:, 'mean_daily_water_temp'].values.reshape(-1, 1)
    air_temp_data.loc[:, 'predicted_daily_max_water_temp'] = max_mean_temp_regr.predict(x2)

    # getting the start and end dates for the data
    start_date = river_flow_data['date'].min()
    end_date = river_flow_data['date'].max()
    list_startdates = pd.date_range(start_date, end_date, freq='YS')

    # Turning the df from long to wide to access by hydrological year
    # todo is there a better way to do this?
    all_hydro_years_df = pd.DataFrame(index=range(1, 367), columns=list_startdates)
    for y in list_startdates:
        l = range(1, len(river_flow_data.loc[river_flow_data.water_year == y, 'flow']) + 1)
        all_hydro_years_df.loc[l, y] = river_flow_data.loc[river_flow_data.water_year == y, 'flow'].values

    temperature_wide_df = pd.DataFrame(index=range(1, 367), columns=list_startdates)
    for x in list_startdates:
        length = range(1, len(air_temp_data.loc[air_temp_data.water_year == x, 'predicted_daily_max_water_temp']) + 1)
        temperature_wide_df.loc[length, x] = air_temp_data.loc[
            air_temp_data.water_year == x, 'predicted_daily_max_water_temp'].values

    # Calculating the median flow for all years
    # One value for the entire dataset
    median_flow = river_flow_data['flow'].median()

    seven_day_avg_df = get_seven_day_avg(river_flow_data)

    # Calculating the ALFs using a nested function
    outdata = pd.DataFrame(index=list_startdates)
    # calc alf
    outdata.loc[:, 'alf'] = seven_day_avg_df.min()

    # Getting the MALF
    # todo needs to be discussed if the reference MALF will change, and therefore whether this needs to change
    outdata.loc[:, 'reference_malf'] = malf = malf_baseline_nat
    outdata.loc[:, 'period_malf'] = outdata['alf'].mean()

    # putting the median in outdata
    outdata.loc[:, 'median'] = median_flow

    # Calculating the days per year spent below MALF
    outdata.loc[:, 'days_below_malf'] = (all_hydro_years_df < malf).sum()

    # getting temperature days
    outdata.loc[:, 'temp_days_above_19'] = (temperature_wide_df > 19).sum()
    outdata.loc[:, 'temp_days_above_21'] = (temperature_wide_df > 21).sum()
    outdata.loc[:, 'temp_days_above_24'] = (temperature_wide_df > 24).sum()

    # consecutive days below malf
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


    # getting malf - alf for each year
    for i, a in outdata.loc[:, 'alf'].items():
        outdata.loc[i, 'anomalies'] = malf - a

    # getting wua for each species for each alf
    for sp in species_limits:
        for k, v in outdata.loc[:, 'alf'].items():
            wua = flow_to_wua(v, sp)
            outdata.loc[k, f'{sp}_wua'] = wua

    # flooding statistics
    # find the maximum (single) flow per year
    outdata.loc[:, 'max_flow'] = all_hydro_years_df.max()
    # find the MAF
    # stats are always referenced to the naturalised baseline maf
    maf = maf_baseline_nat
    outdata.loc[:, 'reference_maf'] = maf
    outdata.loc[:, 'period_maf'] = outdata.loc[:, 'max_flow'].mean()

    # find the flood anomaly
    for i, m in outdata.loc[:, 'max_flow'].items():
        outdata.loc[i, 'flood_anomalies'] = maf - m

    # find the number of days > MAF
    outdata.loc[:, 'days_above_maf'] = (all_hydro_years_df > maf).sum()

    # combination MAF vs MALF stats
    outdata.loc[:, 'maf_times_malf'] = outdata.loc[:, 'days_above_maf'] * outdata.loc[:, 'days_below_malf']

    # KEYNOTE SCORING STARTS HERE
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
    baseline_temperature_days = {'min_temp_days_above_19': 0, 'max_temp_days_above_19': 23,
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

    # flooding scores

    baseline_flood_anomaly = {'min': -977.0379620689654, 'max': 431.5956442310345}
    baseline_days_above_maf = {'min': 0, 'max': 3}
    baseline_maf_and_malf_days = {'min': 0, 'max': 180}

    # days above maf score
    for idx6, value6 in outdata.loc[:, 'days_above_maf'].items():
        min_v5, max_v5 = baseline_days_above_maf['min'], baseline_days_above_maf['max']
        days_score = higher_is_worse(min_v5, max_v5, value6)
        outdata.loc[idx6, 'days_above_maf_score'] = days_score

    # flood anomaly score
    for idx7, value7 in outdata.loc[:, 'flood_anomalies'].items():
        min_v6, max_v6 = baseline_flood_anomaly['min'], baseline_flood_anomaly['max']
        anomalies_score = higher_is_better(min_v6, max_v6, value7)
        outdata.loc[idx7, 'flood_anomalies_score'] = anomalies_score

    # malf days * maf days score
    for idx8, value8 in outdata.loc[:, 'maf_times_malf'].items():
        min_v7, max_v7 = baseline_maf_and_malf_days['min'], baseline_maf_and_malf_days['max']
        score = higher_is_worse(min_v7, max_v7, value8)
        outdata.loc[idx8, 'malf_times_maf_score'] = score

    # calculating weighted score
    for variable, weight in weightings.item():
        if variable in outdata.columns:
            outdata[f'{variable}_weighted'] = outdata[variable] * weight

    return outdata

# todo figure out the best way to input the data for this
def calculate_annual_scores(detailed_scores):
    """A function that calculates the annual scores for each year by averaging all the variable scores for that year"""


    detailed_scores = detailed_scores.set_index('water_year')
    detailed_scores['yearly_avg_score'] = detailed_scores.mean(axis=1)
    detailed_scores['yearly_avg_score'] = round((detailed_scores['yearly_avg_score'] * 2.0)) / 2.0
    detailed_scores['timeseries_avg_score'] = detailed_scores['yearly_avg_score'].mean()
    detailed_scores['rounded_timeseries_avg_score'] = round((detailed_scores['timeseries_avg_score'] * 2.0)) / 2.0

    return detailed_scores


def calculate_ts_scores(detailed_scores):

    """A function that calculates the timeseries scores for the entire dataset"""

    detailed_scores['timeseries_avg_score'] = detailed_scores['yearly_avg_score'].mean()
    detailed_scores['rounded_timeseries_avg_score'] = round((detailed_scores['timeseries_avg_score'] * 2.0)) / 2.0

    return detailed_scores


if __name__ == '__main__':
    air_temp = pd.read_csv(KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info',
                                                                            '2024_test_data.csv',
                                                                            'test_air_temp_data.csv'))
    river_flow = pd.read_csv(
        KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info', '2024_test_data.csv',
                                                         'test_river_flow_data.csv'))
    generate_scores(air_temp, river_flow, weightings=None, flow_limits=None)

pass