"""
 Author: Matt Hanson
 Created: 4/01/2021 9:19 AM
 """
import ksl_env
import os

event_def_dir = ksl_env.shared_drives("Z2003_SLMACC\event_definition/v5")

if not os.path.exists(event_def_dir):
    os.makedirs(event_def_dir)

event_def_path_drive = os.path.join(event_def_dir,
                                    'event_definition_data.csv')  # todo add final event definintions to this repo and update here
event_def_path = os.path.join(os.path.dirname(__file__), 'event_definition_data.csv')
