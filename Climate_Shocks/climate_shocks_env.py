"""
 Author: Matt Hanson
 Created: 4/01/2021 9:19 AM
 """
import ksl_env
import os

event_def_path = os.path.join(os.path.dirname(__file__), 'supporting_data', 'event_definition_data.csv')  # in git repo
supporting_data_dir = os.path.dirname(event_def_path)

storyline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Storylines/storyline_csvs')

temp_output_dir = os.path.join(ksl_env.slmmac_dir_unbacked,'outputs_for_ws')
if not os.path.exists(temp_output_dir):
    os.makedirs(temp_output_dir)

temp_storyline_dir = os.path.join(ksl_env.slmmac_dir_unbacked, 'temp_storyline_files')
if not os.path.exists(temp_storyline_dir):
    os.makedirs(temp_storyline_dir)
