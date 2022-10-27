"""
created Evelyn_Charlesworth 
on: 25/10/2022
"""
"""a script that allows the yearly scores created in ecological_scoring.py to be turned into yearly and then
period average scores"""

import kslcore
import pandas as pd
import numpy as np

def get_avg_score(filename, out_filename):
    """a function that reads in the original spreadsheets and creates an average yearly
    score and then timeseries score based on all the variable scores"""

    pathway = kslcore.KslEnv.shared_gdrive.joinpath(f'Z2003_SLMACC/eco_modelling/stats_info/storyline_data/{filename}.csv')
    df = pd.read_csv(pathway)
    df = df[['water_year', 'longfin_eel_score', "shortfin_eel_score", 'torrent_fish_score', 'common_bully_score',
             'upland_bully_score',	'bluegill_bully_score',	'food_production_score',
             'brown_trout_adult_score',	'chinook_salmon_junior_score',	'diatoms_score',
             'long_filamentous_score', 'short_filamentous_score', 'days_below_malf_score',
             'days_below_flow_lim_score', 'anomalies_score', 'malf_events_greater_7_score',
             'malf_events_greater_14_score', 'malf_events_greater_21_score', 'malf_events_greater_28_score',
             'flow_events_greater_7_score',	'flow_events_greater_14_score',	'flow_events_greater_21_score',
             'flow_events_greater_28_score', 'temp_days_above_19_score','temp_days_above_21_score',
             'temp_days_above_24_score']]

    df = df.set_index('water_year')
    df['yearly_avg_score'] = df.mean(axis=1)
    df['yearly_avg_score'] = round((df['yearly_avg_score'] * 2.0))/ 2.0
    df['timeseries_avg_score'] = df['yearly_avg_score'].mean()
    df['rounded_timeseries_avg_score'] = round((df['timeseries_avg_score'] * 2.0)) / 2.0

    outpath = kslcore.KslEnv.shared_gdrive.joinpath(f'Z2003_SLMACC/eco_modelling/stats_info/storyline_data/{out_filename}.csv')
    df.to_csv(outpath)
    return df

if __name__ == '__main__':
    get_avg_score('measured_2_bad_stats', 'measured_2_bad_scores')