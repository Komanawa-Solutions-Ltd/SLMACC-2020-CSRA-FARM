"""
 Author: Matt Hanson
 Created: 4/01/2021 9:19 AM
 """
import ksl_env
import os

event_def_dir = ksl_env.shared_drives("Z2003_SLMACC\event_definition/v4")

if not os.path.exists(event_def_dir):
    os.makedirs(event_def_dir)

event_def_path = os.path.join(event_def_dir,
                              'annual_event_data_v2.csv')  # todo add final event definintions to this repo and update here
