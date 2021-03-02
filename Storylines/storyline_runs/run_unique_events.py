"""
 Author: Matt Hanson
 Created: 24/02/2021 10:46 AM
 """
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from Storylines.storyline_building_support import make_sampling_options, base_events, default_storyline_time, \
    map_storyline_rest, irrig_season
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp, default_pasture_growth_dir, pgm_log_dir

unique_events_storyline = os.path.join(temp_storyline_dir, 'unique_event_events')
unique_events_pgr_dir = os.path.join(default_pasture_growth_dir, 'unique_event_events')

if not os.path.exists(unique_events_storyline):
    os.makedirs(unique_events_storyline)
if not os.path.exists(unique_events_pgr_dir):
    os.makedirs(unique_events_pgr_dir)


def make_storylines():
    options = make_sampling_options()
    years = default_storyline_time.year
    months = default_storyline_time.month
    base_data = pd.DataFrame(index=default_storyline_time[0:12], columns=['precip_class', 'temp_class', 'rest'])
    base_data.loc[:, 'month'] = months[0:12]
    base_data.loc[:, 'year'] = years[0:12]
    base_data.loc[:, 'use_rest'] = 0.50
    for i in base_data.index:
        t, p, r = base_events[base_data.loc[i, 'month']]
        base_data.loc[i, 'precip_class'] = p
        base_data.loc[i, 'temp_class'] = t

    for m in range(1, 13):
        m_options = options[m]
        for t, p, r in m_options:
            if m in irrig_season:
                rest_vals = [0.5, 0.75, 0.95]
            else:
                rest_vals = [0]
            for rest in rest_vals:
                outdata = deepcopy(base_data)
                idx = outdata.month == m
                outdata.loc[idx, 'precip_class'] = p
                outdata.loc[idx, 'temp_class'] = t
                outdata.loc[idx, 'use_rest'] = rest
                outdata.loc[:, 'rest'] = outdata.loc[:, 'use_rest']
                map_storyline_rest(outdata)
                outdata.to_csv(os.path.join(unique_events_storyline, f'm{m:02d}-{p}-{t}-{int(rest * 100)}.csv'))


def run_pasture_growth():  # todo
    storyline_paths = [os.path.join(unique_events_storyline, e) for e in os.listdir(unique_events_storyline)]
    odirs = [unique_events_pgr_dir for e in storyline_paths]
    nsims = 1000

    run_full_model_mp(storyline_path_mult=storyline_paths,
                      outdir_mult=odirs,
                      nsims_mult=nsims,
                      log_path=os.path.join(pgm_log_dir, 'unique_events'),
                      description_mult='to run each event indivdually',
                      padock_rest_mult=False,
                      save_daily_mult=False,
                      verbose=False)


def extract_data():  # todo
    raise NotImplementedError


if __name__ == '__main__':
    run_pgr = False
    if run_pgr:
        make_storylines()
        run_pasture_growth()
