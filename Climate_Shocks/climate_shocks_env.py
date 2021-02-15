"""
 Author: Matt Hanson
 Created: 4/01/2021 9:19 AM
 """
import ksl_env
import os

event_def_dir = ksl_env.shared_drives("Z2003_SLMACC\event_definition/v5")

if not os.path.exists(event_def_dir):
    os.makedirs(event_def_dir)

event_def_path_drive = os.path.join(event_def_dir, 'event_definition_data.csv') # in google drive
event_def_path = os.path.join(os.path.dirname(__file__), 'supporting_data', 'event_definition_data.csv')  # in git repo

storyline_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Storylines/storyline_csvs')