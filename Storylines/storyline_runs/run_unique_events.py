"""
 Author: Matt Hanson
 Created: 24/02/2021 10:46 AM
 """
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import itertools
import glob
from copy import deepcopy
import project_base
from Storylines.storyline_building_support import make_sampling_options, base_events, default_storyline_time, \
    map_storyline_rest, irrig_season, month_len, default_mode_sites
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
    base_data = pd.DataFrame(index=default_storyline_time[0:24], columns=['precip_class', 'temp_class', 'rest'])
    base_data.loc[:, 'month'] = months[0:24]
    base_data.loc[:, 'year'] = years[0:24]
    base_data.loc[:, 'use_rest'] = 0.50
    for i in base_data.index:
        t, p, r, rp = base_events[base_data.loc[i, 'month']]
        base_data.loc[i, 'precip_class'] = p
        base_data.loc[i, 'temp_class'] = t

    for m in range(1, 13):
        m_options = options[m]
        for t, p, r in m_options:
            if m in irrig_season:
                rest_vals = [0.05, 0.25, 0.5, 0.75, 0.95]
            else:
                rest_vals = [0]
            for rest in rest_vals:
                outdata = deepcopy(base_data)
                idx = outdata.month == m
                outdata.loc[idx, 'precip_class'] = p
                outdata.loc[idx, 'temp_class'] = t
                outdata.loc[idx, 'use_rest'] = rest
                outdata.loc[:, 'rest'] = outdata.loc[:, 'use_rest']
                outdata.loc[:, 'rest_per'] = outdata.loc[:, 'use_rest']
                map_storyline_rest(outdata)
                outdata.to_csv(os.path.join(unique_events_storyline, f'm{m:02d}-{p}-{t}-{int(rest * 100)}.csv'))


def run_pasture_growth():
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
                      verbose=False,
                      mode_sites_mult=default_mode_sites)


def extract_data(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    sms = [f'{s}-{m}' for m, s in default_mode_sites]
    for sm, var, cum in itertools.product(sms,
                                          ['PGRA', 'PGR'],
                                          [True, False]):
        paths = sorted(glob.glob(os.path.join(unique_events_pgr_dir, f'*{sm}.nc')))
        names = [os.path.basename(os.path.splitext(p)[0]) for p in paths]
        outdata = pd.DataFrame(index=range(0, 24), columns=names, dtype=float)
        outdata.index.name = 'months since event month'
        for p, n in zip(paths, names):
            data = nc.Dataset(p)
            months = np.array(data.variables['m_month'])
            idx = np.where(months == int(n.split('-')[0].replace('m', '')))[0][0]
            use_data = np.nanmean(np.array(data.variables[f'm_{var}'][idx:]), axis=1)
            if cum:
                cum_nm = 'monthly_total'
                outdata.loc[range(len(use_data)), n] = use_data * month_len[int(n.split('-')[0].replace('m', ''))]
            else:
                cum_nm = 'daily_total'
                outdata.loc[range(len(use_data)), n] = use_data

            data.close()
        outdata = outdata.round(2).transpose()
        temp = list(outdata.columns.values)
        outdata.loc[:, 'month'] = outdata.index.str.split('-').str[0].str.replace('m', '')
        outdata.loc[:, 'precip'] = outdata.index.str.split('-').str[1]
        outdata.loc[:, 'temp'] = outdata.index.str.split('-').str[2]
        outdata.loc[:, 'rest'] = outdata.index.str.split('-').str[3]

        outdata = outdata.loc[:, ['month',
                                  'precip',
                                  'temp',
                                  'rest'] + temp]
        outdata.to_csv(os.path.join(outdir, f'{sm}-{var}-{cum_nm}-singe_events.csv'))


if __name__ == '__main__':
    # todo need to adjust output data so that it is relative to the two year storyline

    # todo re-run, then need to plot and show
    mk_st = False
    run_pgr = False
    extract = True
    if mk_st:
        make_storylines()
    if run_pgr:
        run_pasture_growth()
    if extract:
        extract_data(os.path.join(ksl_env.slmmac_dir, 'outputs_for_ws', 'norm', 'unique_events_v2'))
