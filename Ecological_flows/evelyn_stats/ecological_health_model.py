"""This is the updated / rewritten version of the Ecological Health model, written on 31/07/24
It uses the ideas and code from ecological_scoring.py and updated_ecological_scoringV2.py
but is updated to be more userfriendly"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from komanawa.kslcore import KslEnv
from Ecological_flows.evelyn_stats.water_temp_monthly import temp_regr
from Ecological_flows.evelyn_stats.max_vs_mean import max_mean_temp_regr

# todo after tested, help write assertions

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


# todo this is wrong, maybe use old function
def calculate_wua(flow: float, species: str, coefficients: Dict[str, list], flow_limits: Dict[str, tuple]) -> float:
    if flow < flow_limits[species][0] or flow > flow_limits[species][1]:
        return 0
    return sum(coef * flow ** i for i, coef in enumerate(coefficients[species]))


def calculate_statistics(flow_data: pd.DataFrame, temp_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # This function will calculate all the required statistics

    # Handling datatypes
    flow_data['datetime'] = pd.to_datetime(flow_data['datetime'], format='%d/%m/%Y')
    temp_data['datetime'] = pd.to_datetime(temp_data['datetime'], format='%d/%m/%Y')
    flow_data.astype({'flow': 'float64'})
    temp_data.astype({'temp': 'float64'})

    # using the regression relationship to get the daily mean water temp from the daily air temp
    # todo how to do this in a way that doesn't call on the original functions/from the waiau csvs
    # help
    x = temp_data.loc[:, 'temp'].values.reshape(-1, 1)
    temp_data.loc[:, 'mean_daily_water_temp'] = temp_regr.predict(x)
    x2 = temp_data.loc[:, 'mean_daily_water_temp'].values.reshape(-1, 1)
    temp_data.loc[:, 'predicted_daily_max_water_temp'] = max_mean_temp_regr.predict(x2)

    # Ensure datetime column is the index
    flow_data = flow_data.set_index('datetime')
    temp_data = temp_data.set_index('datetime')

    # A function to get the hydrological year
    # todo this is wrong
    def get_hydro_year(date):
        if date.month >= 7:
            return date.year + 1
        else:
            return date.year

    # Add hydrological year column
    flow_data['hydro_year'] = flow_data.index.map(get_hydro_year)
    temp_data['hydro_year'] = temp_data.index.map(get_hydro_year)

    # Calculate the statistics
    stats = {}

    # Annual low flow per hydrological year (ALF)
    # todo needs to be done using a 7 day rolling average
    stats['alf'] = flow_data.groupby('hydro_year')['flow'].min()

    # Mean annual low flow (MALF)
    # using the baseline naturalised malf as the comparative malf (as per original scoring)
    malf = 42.2007397

    # Median flow per hydrological year
    stats['median_flow'] = flow_data.groupby('hydro_year')['flow'].median()

    # Number of days below MALF per hydrological year
    stats['days_below_malf'] = flow_data[flow_data['flow'] < malf].groupby('hydro_year').size()

    # Number of days below flow limit per hydrological year
    flow_limit = 50
    stats['days_below_flow_limit'] = flow_data[flow_data['flow'] < flow_limit].groupby('hydro_year').size()

    # Temperature thresholds
    for threshold in [19, 21, 24]:
        stats[f'days_above_{threshold}C'] = temp_data[temp_data['predicted_daily_max_water_temp'] > threshold].groupby('hydro_year').size()

    # MALF events
    def count_events(group, threshold, min_days):
        events = (group < threshold).astype(int).diff().ne(0).cumsum()
        return events[group < threshold].value_counts().ge(min_days).sum()

    for days in [7, 14, 21, 28]:
        stats[f'malf_events_{days}d'] = flow_data.groupby('hydro_year')['flow'].apply(
            lambda x: count_events(x, malf, days))

    # Flow limit events
    for days in [7, 14, 21, 28]:
        stats[f'flow_limit_events_{days}d'] = flow_data.groupby('hydro_year')['flow'].apply(
            lambda x: count_events(x, flow_limit, days))

    # Anomaly
    stats['anomaly'] = malf - stats['alf']

    # Mean annual flood (MAF)
    annual_max_flow = flow_data.groupby('hydro_year')['flow'].max()
    mean_annual_flood = annual_max_flow.mean()

    # Flood anomaly
    stats['flood_anomaly'] = annual_max_flow - mean_annual_flood

    # Days above MAF
    stats['days_above_maf'] = flow_data[flow_data['flow'] > mean_annual_flood].groupby('hydro_year').size()

    # MAF times MALF
    stats['maf_malf_product'] = stats['days_above_maf'] * stats['days_below_malf']
    # replacing any maf_malf_products that are NaN with 0
    stats['maf_malf_product'] = stats['maf_malf_product'].fillna(0)

    # implementing the WUA calculation for each species
    for species in species_coeffs.keys():
        stats[f'{species}_wua'] = stats['alf'].apply(
            lambda flow: calculate_wua(flow, species, species_coeffs, species_limits))

    return stats


def score_variable(value: float, min_value: float, max_value: float, is_higher_better: bool) -> float:
    # This function calculates the score for a single variable
    if min_value == max_value:
        return 0
    score = (value - min_value) / (max_value - min_value)
    if not is_higher_better:
        score = score * -1
    return max(min(score * 6 - 3, 3), -3)


def calculate_scores(stats: Dict[str, pd.DataFrame], baseline_min_max: Dict[str, float],
                     weightings: Optional[Dict[str, float]] = None) -> Dict[str, pd.DataFrame]:
    # This function calculates scores for all variables

    scores = {}
    for var, data in stats.items():
        if var == 'alf':
            continue
        elif var == 'median_flow':
            continue
        min_value = baseline_min_max[f'{var}_min']
        max_value = baseline_min_max[f'{var}_max']
        is_higher_better = var in ["longfin_eel_<300_wua", "torrent_fish_wua", "brown_trout_adult_wua", "diatoms_wua",
                                   "long_filamentous_wua", "anomaly"]
        scores[var] = data.apply(lambda x: score_variable(x, min_value, max_value, is_higher_better))

        if weightings is None:
            weightings = {'longfin_eel_<300_wua': 1, 'torrent_fish_wua': 1, 'brown_trout_adult_wua': 1,
                          'diatoms_wua': 1,
                          'long_filamentous_wua': 1,
                          'days_below_malf': 1, 'days_below_flow_limit': 1, 'anomaly': 1,
                          'malf_events_7d': 1, 'malf_events_14d': 1,
                          'malf_events_21d': 1, 'malf_events_28d': 1,
                          'flow_limit_events_7d': 1, 'flow_limit_events_14d': 1,
                          'flow_limit_events_21d': 1, 'flow_limit_events_28d': 1,
                          'days_above_19C': 1, 'days_above_21C': 1,
                          'days_above_24C': 1,
                          'days_above_maf': 1, 'flood_anomaly': 1, 'maf_malf_product': 1}
        for variable, weights in weightings.items():
            if var in variable:
                scores[f'{var}'] = scores[var] * weights

    return scores


def calculate_yearly_scores(scores: Dict[str, pd.DataFrame]) -> pd.DataFrame():
    # This function calculates average yearly scores
    return pd.DataFrame(scores).mean(axis=1)


def calculate_timeseries_score(yearly_scores: pd.DataFrame) -> float:
    # This function calculates the average timeseries score
    return yearly_scores.mean()


def analyse_stream_health(flow_data: pd.DataFrame, temp_data: pd.DataFrame,
                          baseline_min_max: Dict[str, float],
                          weightings: Optional[Dict[str, Dict[str, float]]] = None) -> Dict:
    # This is the wrapper function that calls all other functions
    # Calculate statistics
    stats = calculate_statistics(flow_data, temp_data)

    # Calculate scores
    scores = calculate_scores(stats, baseline_min_max, weightings)

    # Calculate yearly scores
    yearly_scores = calculate_yearly_scores(scores)

    # Calculate timeseries score
    timeseries_score = calculate_timeseries_score(yearly_scores)

    total_scores = {
        'statistics': stats,
        'scores': scores,
        'yearly_scores': yearly_scores,
        'timeseries_score': timeseries_score
    }
    return total_scores

def get_expert_weightings(expert_name):
    """
    :param expert_name: str, name of the expert
    :return: dict, weightings for each ecological flow component
    """
    # todo check how we want to name the experts - keep anonymous or use real names?
    # keynote the median weightings for each variable have been used
    if expert_name == 'duncan_grey':
        return {'longfin_eel_<300_wua': 2, 'torrent_fish_wua': 3.5, 'brown_trout_adult_wua': 2.5,
                'diatoms_wua': 2.5,
                'long_filamentous_wua': 4.5,
                'days_below_malf': 4, 'days_below_flow_limit': 3.5, 'anomaly': 4,
                'malf_events_7d': 2.5, 'malf_events_14d': 3.5,
                'malf_events_21d': 4, 'malf_events_28d': 4,
                'flow_limit_events_7d': 2.5, 'flow_limit_events_14d': 2.5,
                'flow_limit_events_21d': 3.5, 'flow_limit_events_28d': 4,
                'days_above_19C': 2.5, 'days_above_21C': 3, 'days_above_24C': 3.5,
                'days_above_maf': 3, 'flood_anomaly': 3, 'maf_malf_product': 3.5}
    elif expert_name == 'rich_allibone':
        return {'longfin_eel_<300_wua': 1.5, 'torrent_fish_wua': 3.5, 'brown_trout_adult_wua': 1.5,
                'diatoms_wua': 4,
                'long_filamentous_wua': 0,
                'days_below_malf': 2, 'days_below_flow_limit': 2, 'anomaly': 0,
                'malf_events_7d': 1.5, 'malf_events_14d': 2,
                'malf_events_21d': 3, 'malf_events_28d': 3.5,
                'flow_limit_events_7d': 0.75, 'flow_limit_events_14d': 0.75,
                'flow_limit_events_21d': 0.75, 'flow_limit_events_28d': 1,
                'days_above_19C': 2.5, 'days_above_21C': 3.25, 'days_above_24C': 4.25,
                'days_above_maf': 3, 'flood_anomaly': 0, 'maf_malf_product': 0}
    elif expert_name == 'greg_burrell':
        return {'longfin_eel_<300_wua': 1, 'torrent_fish_wua': 1, 'brown_trout_adult_wua': 1,
                'diatoms_wua': 1,
                'long_filamentous_wua': -1,
                'days_below_malf': 4, 'days_below_flow_limit': 3.5, 'anomaly': 4,
                'malf_events_7d': 3.5, 'malf_events_14d': 3.5,
                'malf_events_21d': 3.5, 'malf_events_28d': 3.5,
                'flow_limit_events_7d': 3.5, 'flow_limit_events_14d': 3.5,
                'flow_limit_events_21d': 3.5, 'flow_limit_events_28d': 3.5,
                'days_above_19C': 4, 'days_above_21C': 4, 'days_above_24C': 4,
                'days_above_maf': 4, 'flood_anomaly': 4, 'maf_malf_product': 4}
    elif expert_name == 'adrian_meredith':
        return {'longfin_eel_<300_wua': 3.5, 'torrent_fish_wua': 4.5, 'brown_trout_adult_wua': 3.5,
                'diatoms_wua': 4,
                'long_filamentous_wua': -3,
                'days_below_malf': 4, 'days_below_flow_limit': 4.5, 'anomaly': 2,
                'malf_events_7d': 4.5, 'malf_events_14d': 4,
                'malf_events_21d': 4.5, 'malf_events_28d': 5,
                'flow_limit_events_7d': 4.5, 'flow_limit_events_14d': 4,
                'flow_limit_events_21d': 4.5, 'flow_limit_events_28d': 4,
                'days_above_19C': 3, 'days_above_21C': 4, 'days_above_24C': 4,
                'days_above_maf': 2, 'flood_anomaly': 3, 'maf_malf_product': 1.5}
    elif expert_name == 'pooled':
        return {'longfin_eel_<300_wua': 2, 'torrent_fish_wua': 3.125, 'brown_trout_adult_wua': 2.125,
                        'diatoms_wua': 2.875,
                        'long_filamentous_wua': 0.125,
                        'days_below_malf': 3.5, 'days_below_flow_limit': 3.375, 'anomaly': 2.5,
                        'malf_events_7d': 3, 'malf_events_14d': 3.25,
                        'malf_events_21d': 3.75, 'malf_events_28d': 4,
                        'flow_limit_events_7d': 2.8125, 'flow_limit_events_14d': 2.6875,
                        'flow_limit_events_21d': 3.0625, 'flow_limit_events_28d': 3.125,
                        'days_above_19C': 2.75, 'days_above_21C': 3.5625, 'days_above_24C': 3.9375,
                        'days_above_maf': 3, 'flood_anomaly': 2.5, 'maf_malf_product': 2.25}
    else:
        raise ValueError(f'invalid expert name {expert_name}')


def get_baseline_min_max():
    baseline_min_max = {'malf': 42.2007397, 'maf': 991.5673849310346, "longfin_eel_<300_wua_max": 146,
                      "torrent_fish_wua_max": 71,
                      "brown_trout_adult_wua_max": 19, "diatoms_wua_max": 0.28,
                      "long_filamentous_wua_max": 0.31, "longfin_eel_<300_wua_min": 426, "torrent_fish_wua_min": 395,
                      "brown_trout_adult_wua_min": 25, "diatoms_wua_min": 0.38, "long_filamentous_wua_min": 0.39,
                      'days_below_malf_min': 0, 'days_below_malf_max': 70,
                      'days_below_flow_limit_min': 0, 'days_below_flow_limit_max': 108, 'anomaly_min': -18.20,
                      'anomaly_max': 16.15, 'malf_events_7d_min': 0, 'malf_events_7d_max': 4,
                      'malf_events_14d_min': 0, 'malf_events_14d_max': 2,
                      'malf_events_21d_min': 0, 'malf_events_21d_max': 1,
                      'malf_events_28d_min': 0, 'malf_events_28d_max': 1,
                      'flow_limit_events_7d_min': 0, 'flow_limit_events_7d_max': 6,
                      'flow_limit_events_14d_min': 0, 'flow_limit_events_14d_max': 4,
                      'flow_limit_events_21d_min': 0, 'flow_limit_events_21d_max': 2,
                      'flow_limit_events_28d_min': 0, 'flow_limit_events_28d_max': 1, 'days_above_19C_min': 0,
                      'days_above_19C_max': 23,
                      'days_above_21C_min': 0, 'days_above_21C_max': 3,
                      'days_above_24C_min': 0, 'days_above_24C_max': 1, 'flood_anomaly_min': -977.0379620689654,
                      'flood_anomaly_max': 431.5956442310345,
                      'days_above_maf_min': 0, 'days_above_maf_max': 3, 'maf_malf_product_min': 0,
                      'maf_malf_product_max': 180}
    return baseline_min_max

if __name__ == '__main__':
    # Example usage
    flow_data = pd.read_csv(
        KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info', '2024_test_data',
                                                         'measured_flow_data_storyline_data_2bad_2024_test.csv'))
    temp_data = pd.read_csv(KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info',
                                                                             '2024_test_data',
                                                                     'temperature_data_storylines_2024_test.csv'))
    #stats = calculate_statistics(flow_data, temp_data)
    #baseline_stats = get_baseline_stats()
    #
    #scores = calculate_scores(stats, baseline_stats, weightings=get_expert_weightings('duncan_grey'))
    #yearly_scores = calculate_yearly_scores(scores)
    #timeseries_score = calculate_timeseries_score(yearly_scores)

    test = analyse_stream_health(flow_data, temp_data, baseline_min_max=get_baseline_min_max(), weightings=get_expert_weightings('pooled'))
    pass





