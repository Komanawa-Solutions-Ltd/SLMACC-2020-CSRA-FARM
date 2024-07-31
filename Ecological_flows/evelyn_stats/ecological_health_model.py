"""This is the updated / rewritten version of the Ecological Health model, written on 31/07/24
It uses the ideas and code from ecological_scoring.py and updated_ecological_scoringV2.py
but is updated to be more userfriendly"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from komanawa.kslcore import KslEnv
from Ecological_flows.evelyn_stats.water_temp_monthly import temp_regr
from Ecological_flows.evelyn_stats.max_vs_mean import max_mean_temp_regr

# todo after tested, ask Claude to help write assertions

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


def calculate_wua(flow: float, species: str, coefficients: Dict[str, list], flow_limits: Dict[str, tuple]) -> float:
    if flow < flow_limits[species][0] or flow > flow_limits[species][1]:
        return 0
    return sum(coef * flow ** i for i, coef in enumerate(coefficients[species]))


def calculate_statistics(flow_data: pd.DataFrame, temp_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # This function will calculate all the required statistics

    # Handling datatypes
    flow_data['datetime'] = pd.to_datetime(flow_data['datetime'])
    temp_data['datetime'] = pd.to_datetime(temp_data['datetime'])
    flow_data.astype({'flow': 'float64'})
    temp_data.astype({'temp': 'float64'})


    # using the regression relationship to get the daily mean water temp from the daily air temp
    x = temp_data.loc[:, 'temp'].values.reshape(-1, 1)
    temp_data.loc[:, 'mean_daily_water_temp'] = temp_regr.predict(x)
    x2 = temp_data.loc[:, 'mean_daily_water_temp'].values.reshape(-1, 1)
    temp_data.loc[:, 'predicted_daily_max_water_temp'] = max_mean_temp_regr.predict(x2)

    # Ensure datetime column is the index
    flow_data = flow_data.set_index('datetime')
    temp_data = temp_data.set_index('datetime')

    # A function to get the hydrological year
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
    stats['alf'] = flow_data.groupby('hydro_year')['flow'].min()

    # Mean annual low flow (MALF)
    malf = stats['alf'].mean()

    # Median flow per hydrological year
    stats['median_flow'] = flow_data.groupby('hydro_year')['flow'].median()

    # Number of days below MALF per hydrological year
    stats['days_below_malf'] = flow_data[flow_data['flow'] < malf].groupby('hydro_year').size()

    # Temperature thresholds
    for threshold in [19, 21, 24]:
        stats[f'days_above_{threshold}C'] = temp_data[temp_data['predicted_daily_max_water_temp'] > threshold].groupby('hydro_year').size()

    # MALF events
    def count_events(group, threshold, min_days):
        events = (group < threshold).astype(int).diff().ne(0).cumsum()
        return events[group < threshold].value_counts().ge(min_days).sum()

    for days in [2, 7, 14, 21, 28]:
        stats[f'malf_events_{days}d'] = flow_data.groupby('hydro_year')['flow'].apply(
            lambda x: count_events(x, malf, days))

    # Flow limit events
    flow_limit = 50
    for days in [2, 7, 14, 21, 28]:
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
    stats['maf_malf_product'] = mean_annual_flood * malf

    # implementing the WUA calculation for ach species
    for species in species_coeffs.keys():
        stats[f'{species}_wua'] = flow_data.groupby('hydro_year')['flow'].apply(
            lambda flows: flows.apply(lambda flow: calculate_wua(flow, species, species_coeffs, species_limits))
        )

    return stats

def score_variable(value: float, baseline: float, is_higher_better: bool) -> float:
    # This function calculates the score for a single variable
    # todo check this gives the same scores as the old function, otherwise update with old function
    if is_higher_better:
        score = (value - baseline) / baseline
    else:
        score = (baseline - value) / baseline

    return max(min(score, 3) -3)


def calculate_scores(stats: Dict[str, pd.DataFrame], baseline_stats: Dict[str, pd.DataFrame],
                     weightings: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, pd.DataFrame]:
    # This function calculates scores for all variables

    scores = {}
    for var, data in stats.items():
        baseline = baseline_stats[var]
        is_higher_better = var in ['median_flow', 'days_above_19C', 'days_above_21C', 'days_above_24C', 'maf_malf_product'] # todo edit these
        scores[var] = data.apply(lambda x: score_variable(x, baseline[x.name], is_higher_better))

        if weightings is not None:
            for expert, weights in weightings.items():
                if var in weights:
                    scores[f'{var}_{expert}'] = scores[var] * weights[var]

    return scores


def calculate_yearly_scores(scores: Dict[str, pd.DataFrame]) -> pd.DataFrame():
    # This function calculates average yearly scores
    return pd.DataFrame(scores).mean(axis=1)


def calculate_timeseries_score(yearly_scores: pd.Dataframe) -> float:
    # This function calculates the average timeseries score
    return yearly_scores.mean()


def analyse_stream_health(flow_data: pd.DataFrame, temp_data: pd.DataFrame,
                          baseline_flow_data: pd.DataFrame, baseline_temp_data: pd.DataFrame,
                          weightings: Optional[Dict[str, Dict[str, float]]] = None) -> Dict:
    # This is the wrapper function that calls all other functions
    # Calculate statistics
    stats = calculate_statistics(flow_data, temp_data)
    baseline_stats = calculate_statistics(baseline_flow_data, baseline_temp_data)

    # Calculate scores
    scores = calculate_scores(stats, baseline_stats, weightings)

    # Calculate yearly scores
    yearly_scores = calculate_yearly_scores(scores)

    # Calculate timeseries score
    timeseries_score = calculate_timeseries_score(yearly_scores)

    return {
        'statistics': stats,
        'scores': scores,
        'yearly_scores': yearly_scores,
        'timeseries_score': timeseries_score
    }


if __name__ == '__main__':
    # Example usage
    flow_data = pd.read_csv(
        KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info', '2024_test_data.csv',
                                                         'test_river_flow_data.csv'), parse_dates=['datetime'],
        date_parser=lambda x: pd.to_datetime(x, format='%dd/%mm/%YYYY'))
    temp_data = pd.read_csv(KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info',
                                                                             '2024_test_data.csv',
                                                                             'test_air_temp_data.csv'),
                            parse_dates=['datetime'], date_parser=lambda x: pd.to_datetime(x, format='%dd/%mm/%YYYY'))
    stats = calculate_statistics(flow_data, temp_data)
    pass
    # todo make sure the correct baselines are provided
    #baseline_flow_data = pd.read_csv('baseline_flow_data.csv', parse_dates=['datetime'])
    #baseline_temp_data = pd.read_csv('baseline_temp_data.csv', parse_dates=['datetime'])



