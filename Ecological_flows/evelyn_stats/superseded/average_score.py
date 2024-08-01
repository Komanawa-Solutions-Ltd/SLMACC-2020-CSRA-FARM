"""
created Evelyn_Charlesworth 
on: 25/10/2022
"""
"""a script that allows the yearly scores created in ecological_scoring.py to be turned into yearly and then
period average scores"""


from komanawa.kslcore import KslEnv
import pandas as pd


def get_avg_score(filename, out_filename):
    """a function that reads in the original spreadsheets and creates an average yearly
    score and then timeseries score based on all the variable scores"""

    pathway = KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'workshop_material',
                                                               'test_scenario_scores', f'{filename}.csv')
    df = pd.read_csv(pathway)
    df = df[['water_year', 'longfin_eel_<300_score', 'torrent_fish_score',
             'brown_trout_adult_score',	'diatoms_score',
             'long_filamentous_score', 'days_below_malf_score',
             'days_below_flow_lim_score', 'anomalies_score', 'malf_events_greater_7_score',
             'malf_events_greater_14_score', 'malf_events_greater_21_score', 'malf_events_greater_28_score',
             'flow_events_greater_7_score',	'flow_events_greater_14_score',	'flow_events_greater_21_score',
             'flow_events_greater_28_score', 'temp_days_above_19_score','temp_days_above_21_score',
             'temp_days_above_24_score', 'days_above_maf_score', 'flood_anomalies_score', 'malf_times_maf_score']]

    df = df.set_index('water_year')
    df['yearly_avg_score'] = df.mean(axis=1)
    df['yearly_avg_score'] = round((df['yearly_avg_score'] * 2.0))/ 2.0
    df['timeseries_avg_score'] = df['yearly_avg_score'].mean()
    df['rounded_timeseries_avg_score'] = round((df['timeseries_avg_score'] * 2.0)) / 2.0

    outpath = KslEnv.shared_drive('Z20002SLM_SLMACC').joinpath('eco_modelling', 'workshop_material',
                                                               'test_scenario_scores', f'{out_filename}.csv')
    df.to_csv(outpath)
    return df


if __name__ == '__main__':
    get_avg_score('measured_2_bad_stats_test', 'measured_2_bad_scores_test')