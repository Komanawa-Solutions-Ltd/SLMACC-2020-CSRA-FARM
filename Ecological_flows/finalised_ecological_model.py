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
        return {'longfin_eel_<300_score': 2, 'torrent_fish_score': 3.5, 'brown_trout_adult_score': 2.5,
                'diatoms_score': 2.5,
                'long_filamentous_score': 4.5,
                'days_below_malf_score': 4, 'days_below_flow_lim_score': 3.5, 'anomalies_score': 4,
                'malf_events_greater_7_score': 2.5, 'malf_events_greater_14_score': 3.5,
                'malf_events_greater_21_score': 4, 'malf_events_greater_28_score': 4,
                'flow_events_greater_7_score': 2.5, 'flow_events_greater_14_score': 2.5,
                'flow_events_greater_21_score': 3.5, 'flow_events_greater_28_score': 4,
                'temp_days_above_19_score': 2.5, 'temp_days_above_21_score': 3, 'temp_days_above_24_score': 3.5,
                'days_above_maf_score': 3, 'flood_anomalies_score': 3, 'malf_times_maf_score': 3.5}
    elif expert_name == 'rich_allibone':
        return {'longfin_eel_<300_score': 1.5, 'torrent_fish_score': 3.5, 'brown_trout_adult_score': 1.5,
                'diatoms_score': 4,
                'long_filamentous_score': 0,
                'days_below_malf_score': 2, 'days_below_flow_lim_score': 2, 'anomalies_score': 0,
                'malf_events_greater_7_score': 1.5, 'malf_events_greater_14_score': 2,
                'malf_events_greater_21_score': 3, 'malf_events_greater_28_score': 3.5,
                'flow_events_greater_7_score': 0.75, 'flow_events_greater_14_score': 0.75,
                'flow_events_greater_21_score': 0.75, 'flow_events_greater_28_score': 1,
                'temp_days_above_19_score': 2.5, 'temp_days_above_21_score': 3.25, 'temp_days_above_24_score': 4.25,
                'days_above_maf_score': 3, 'flood_anomalies_score': 0, 'malf_times_maf_score': 0}
    elif expert_name == 'greg_burrell':
        return {'longfin_eel_<300_score': 1, 'torrent_fish_score': 1, 'brown_trout_adult_score': 1, 'diatoms_score': 1,
                'long_filamentous_score': -1,
                'days_below_malf_score': 4, 'days_below_flow_lim_score': 3.5, 'anomalies_score': 4,
                'malf_events_greater_7_score': 3.5, 'malf_events_greater_14_score': 3.5,
                'malf_events_greater_21_score': 3.5, 'malf_events_greater_28_score': 3.5,
                'flow_events_greater_7_score': 3.5, 'flow_events_greater_14_score': 3.5,
                'flow_events_greater_21_score': 3.5, 'flow_events_greater_28_score': 3.5,
                'temp_days_above_19_score': 4, 'temp_days_above_21_score': 4, 'temp_days_above_24_score': 4,
                'days_above_maf_score': 4, 'flood_anomalies_score': 4, 'malf_times_maf_score': 4}
    elif expert_name == 'adrian_meredith':
        return {'longfin_eel_<300_score': 3.5, 'torrent_fish_score': 4.5, 'brown_trout_adult_score': 3.5,
                'diatoms_score': 4,
                'long_filamentous_score': -3,
                'days_below_malf_score': 4, 'days_below_flow_lim_score': 4.5, 'anomalies_score': 2,
                'malf_events_greater_7_score': 4.5, 'malf_events_greater_14_score': 4,
                'malf_events_greater_21_score': 4.5, 'malf_events_greater_28_score': 5,
                'flow_events_greater_7_score': 4.5, 'flow_events_greater_14_score': 4,
                'flow_events_greater_21_score': 4.5, 'flow_events_greater_28_score': 4,
                'temp_days_above_19_score': 3, 'temp_days_above_21_score': 4, 'temp_days_above_24_score': 4,
                'days_above_maf_score': 2, 'flood_anomalies_score': 3, 'malf_times_maf_score': 1.5}
    elif expert_name == 'pooled':
        return {'longfin_eel_<300_score': 2, 'torrent_fish_score': 3.125, 'brown_trout_adult_score': 2.125,
                'diatoms_score': 2.875,
                'long_filamentous_score': 0.125,
                'days_below_malf_score': 3.5, 'days_below_flow_lim_score': 3.375, 'anomalies_score': 2.5,
                'malf_events_greater_7_score': 3, 'malf_events_greater_14_score': 3.25,
                'malf_events_greater_21_score': 3.75, 'malf_events_greater_28_score': 4,
                'flow_events_greater_7_score': 2.8125, 'flow_events_greater_14_score': 2.6875,
                'flow_events_greater_21_score': 3.0625, 'flow_events_greater_28_score': 3.125,
                'temp_days_above_19_score': 2.75, 'temp_days_above_21_score': 3.5625,
                'temp_days_above_24_score': 3.9375,
                'days_above_maf_score': 3, 'flood_anomalies_score': 2.5, 'malf_times_maf_score': 2.25}
    else:
        raise ValueError(f'invalid expert name {expert_name}')


def run_eco_model(air_temp, r_flow, weightings=None):
    """
    :param air_temp: air temperature timeseries in degrees C, daily data
    :param r_flow: river flow timseries in m3/s, daily data
    :param weightings: dict of weightings for each ecological flow component
    :return: ts_score, annual_scores, detailed_scores
    """
    # todo where does this go? in the run_eco_model or in the generate_scores?
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

    detailed_scores = generate_scores(air_temp, r_flow, weightings)

    # todo check data structure for air_temp and r_flow
    # one column should be datetime data, other should be the flow in m3/d or the temp in degrees C
    # todo write some assertions

    annual_scores = calculate_annual_scores(detailed_scores)
    ts_scores = calculate_ts_scores(annual_scores)

    return ts_scores, annual_scores, detailed_scores

if __name__ == '__main__':
    air_temp = None
    r_flow = None
    weightings = get_expert_weightings('expert1')
    out = run_eco_model(air_temp, r_flow, weightings)