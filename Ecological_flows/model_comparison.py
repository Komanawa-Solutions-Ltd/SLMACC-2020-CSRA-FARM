"""
created matt_dumont 
on: 8/9/24
"""

from Ecological_flows.EcologicalHealthModel import EcoHealthModel
from Ecological_flows.evelyn_stats.ecological_health_model import analyse_stream_health, get_expert_weightings, \
    get_baseline_min_max
from komanawa.kslcore import KslEnv
import pandas as pd
import numpy as np

if __name__ == '__main__':
    flow_data = pd.read_csv(
        KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info', '2024_test_data',
                                                         'measured_flow_data_storyline_data_2bad_2024_test.csv'))
    temp_data = pd.read_csv(KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'stats_info',
                                                                             '2024_test_data',
                                                                             'temperature_data_storylines_2024_test.csv'))
    temp_data['date'] = pd.to_datetime(temp_data['datetime'], format='%d/%m/%Y')
    temp_data = temp_data.loc[temp_data['date'] >= '2018-07-01']
    flow_data['date'] = pd.to_datetime(flow_data['datetime'], format='%d/%m/%Y')
    flow_data = flow_data.loc[flow_data['date'] >= '2018-07-01']

    new_model = EcoHealthModel(flow_data['date'].values, flow_data['flow'].values[:, np.newaxis],
                               temp_data['temp'].values[:, np.newaxis],
                               weighting='pooled', baseline='historical')

    old_model = analyse_stream_health(flow_data, temp_data, baseline_min_max=get_baseline_min_max(),
                                      weightings=get_expert_weightings('pooled'))

    old_stats = old_model['statistics']
    new_stats = new_model.water_year_datasets
    for key in old_stats:
        old = old_stats[key]
        key = key.replace('malf_events_7d', 'malf_events_07d')
        key = key.replace('flow_limit_events_7d', 'flow_limit_events_07d')
        key = key.replace('flood_anomaly', 'maf_anomaly')
        key = key.replace('<', 'lt_')
        if key == 'anomaly':
            key = 'malf_anomaly'
        new = new_stats[key]
        if old.empty:
            old = pd.Series([0])
        if old.values[0] != new.values[0, 0]:
            print(key)
            print('old', old.values[0])
            print('new', new.values[0, 0])
            print('')
