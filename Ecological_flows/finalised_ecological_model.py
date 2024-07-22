"""A script that runs the ecological model, allowing for different
expert weightings to be applied to the ecological flow components."""


def get_expert_weightings(expert_name):
    """
    :param expert_name: str, name of the expert
    :return: dict, weightings for each ecological flow component
    """
    # todo check how we want to name the experts - keep anonymous or use real names?
    # keynote the median weightings for each variable have been used
    if expert_name == 'duncan_grey':
        return {'longfin_eel': 2, 'torrent_fish': 3.5, 'brown_trout': 2.5, 'diatoms': 2.5, 'long_filamentous': 4.5,
                'days_below_malf': 4, 'days_below_flow_limit': 3.5, 'anomaly': 4,
                'malf_events_greater_7': 2.5, 'malf_events_greater_14': 3.5,
                'malf_events_greater_21': 4, 'malf_events_greater_28': 4,
                'flow_events_greater_7': 2.5, 'flow_events_greater_14': 2.5,
                'flow_events_greater_21': 3.5, 'flow_events_greater_28': 4,
                'temp_days_above_19': 2.5, 'temp_days_above_21': 3, 'temp_days_above_24': 3.5,
                'days_above_maf': 3, 'flood_anomalies': 3, 'malf_times_maf': 3.5}
    elif expert_name == 'rich_allibone':
        return {'longfin_eel': 1.5, 'torrent_fish': 3.5, 'brown_trout': 1.5, 'diatoms': 4, 'long_filamentous': 0,
                'days_below_malf': 2, 'days_below_flow_limit': 2, 'anomaly': 0,
                'malf_events_greater_7': 1.5, 'malf_events_greater_14': 2,
                'malf_events_greater_21': 3, 'malf_events_greater_28': 3.5,
                'flow_events_greater_7': 0.75, 'flow_events_greater_14': 0.75,
                'flow_events_greater_21': 0.75, 'flow_events_greater_28': 1,
                'temp_days_above_19': 2.5, 'temp_days_above_21': 3.25, 'temp_days_above_24': 4.25,
                'days_above_maf': 3, 'flood_anomalies': 0, 'malf_times_maf': 0}
    elif expert_name == 'greg_burrell':
        return {'longfin_eel': 1, 'torrent_fish': 1, 'brown_trout': 1, 'diatoms': 1, 'long_filamentous': -1,
                'days_below_malf': 4, 'days_below_flow_limit': 3.5, 'anomaly': 4,
                'malf_events_greater_7': 3.5, 'malf_events_greater_14': 3.5,
                'malf_events_greater_21': 3.5, 'malf_events_greater_28': 3.5,
                'flow_events_greater_7': 3.5, 'flow_events_greater_14': 3.5,
                'flow_events_greater_21': 3.5, 'flow_events_greater_28': 3.5,
                'temp_days_above_19': 4, 'temp_days_above_21': 4, 'temp_days_above_24': 4,
                'days_above_maf': 4, 'flood_anomalies': 4, 'malf_times_maf': 4}
    elif expert_name == 'adrian_meredith':
        return {'longfin_eel': 3.5, 'torrent_fish': 4.5, 'brown_trout': 3.5, 'diatoms': 4, 'long_filamentous': -3,
                'days_below_malf': 4, 'days_below_flow_limit': 4.5, 'anomaly': 2,
                'malf_events_greater_7': 4.5, 'malf_events_greater_14': 4,
                'malf_events_greater_21': 4.5, 'malf_events_greater_28': 5,
                'flow_events_greater_7': 4.5, 'flow_events_greater_14': 4,
                'flow_events_greater_21': 4.5, 'flow_events_greater_28': 4,
                'temp_days_above_19': 3, 'temp_days_above_21': 4, 'temp_days_above_24': 4,
                'days_above_maf': 2, 'flood_anomalies': 3, 'malf_times_maf': 1.5}
    elif expert_name == 'pooled':
        return {'longfin_eel': 2, 'torrent_fish': 3.125, 'brown_trout': 2.125, 'diatoms': 2.875, 'long_filamentous': 0.125,
                'days_below_malf': 3.5, 'days_below_flow_limit': 3.375, 'anomaly': 2.5,
                'malf_events_greater_7': 3, 'malf_events_greater_14': 3.25,
                'malf_events_greater_21': 3.75, 'malf_events_greater_28': 4,
                'flow_events_greater_7': 2.8125, 'flow_events_greater_14': 2.6875,
                'flow_events_greater_21': 3.0625, 'flow_events_greater_28': 3.125,
                'temp_days_above_19': 2.75, 'temp_days_above_21': 3.5625, 'temp_days_above_24': 3.9375,
                'days_above_maf': 3, 'flood_anomalies': 2.5, 'malf_times_maf': 2.25}
    else:
        raise ValueError(f'invalid expert name {expert_name}')


def run_eco_model(air_temp, r_flow, weightings=None):
    """
    :param air_temp: air temperature timeseries in degrees C, daily data
    :param r_flow: river flow timseries in m3/s, daily data
    :param weightings: dict of weightings for each ecological flow component
    :return: ts_score, annual_scores, detailed_scores
    """
    if weightings is None:
        weightings = {'longfin_eel': 1, 'torrent_fish': 1, 'brown_trout': 1, 'diatoms': 1, 'long_filamentous': 1,
                      'days_below_malf': 1, 'days_below_flow_limit': 1, 'anomaly': 1,
                      'malf_events_greater_7': 1, 'malf_events_greater_14': 1,
                      'malf_events_greater_21': 1, 'malf_events_greater_28': 1,
                      'flow_events_greater_7': 1, 'flow_events_greater_14': 1,
                      'flow_events_greater_21': 1, 'flow_events_greater_28': 1,
                      'temp_days_above_19': 1, 'temp_days_above_21': 1, 'temp_days_above_24': 1,
                      'days_above_maf': 1, 'flood_anomalies': 1, 'malf_times_maf': 1}
    else:
        raise NotImplementedError

    detailed_scores = generate_scores(air_temp, r_flow, weightings)
    # todo apply weightings within the detailed scores function, or return scores and then weight them??
    # todo I think want to do as here, where the weighting is input into the run_eco_model function, but the weighting
    # todo calculation is done within the generate_scores function

    # todo check data structure for air_temp and r_flow
    # one column should be datetime data, other should be the flow in m3/d or the temp in degrees C
    # todo write some assertions

    annual_scores = calculate_annual_scores(detailed_scores)
    ts_scores = calculate_ts_scores(annual_scores)

    return ts_scores, annual_scores, detailed_scores
