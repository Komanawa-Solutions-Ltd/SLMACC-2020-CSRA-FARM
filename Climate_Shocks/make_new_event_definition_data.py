"""
 Author: Matt Hanson
 Created: 23/02/2021 10:27 AM
 """
import pandas as pd
import numpy as np
import os
import ksl_env
import subprocess
import sys
from Climate_Shocks.note_worthy_events.rough_stats import make_data
from Climate_Shocks.climate_shocks_env import event_def_path, supporting_data_dir
from Climate_Shocks.note_worthy_events.final_event_recurance import get_org_data
from Storylines.check_storyline import get_past_event_frequency, get_acceptable_events

if __name__ == '__main__':  # todo check outputs
    prev_event_path = ksl_env.shared_drives(r"Z2003_SLMACC\event_definition\v5_detrend\detrend_event_data.csv")
    event_def_dir = ksl_env.shared_drives(r"Z2003_SLMACC\event_definition/v6_detrend")
    vcsn_version = 'detrended2'

    root_dir = os.path.dirname(os.path.dirname(__file__))
    # run inital event recurance
    print('running inital for vcsn: {} and event_def_dir: {}'.format(vcsn_version, event_def_dir))
    inital = os.path.join(root_dir, 'Climate_Shocks/note_worthy_events/inital_event_recurance.py')
    result = subprocess.run([sys.executable, inital, event_def_dir, vcsn_version],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    # make event data
    print('making final event_data for vcsn: {} and event_def_dir: {}'.format(vcsn_version, event_def_dir))

    temp = make_data(get_org_data(event_def_dir), save=True,
                     save_paths=[event_def_path, os.path.join(event_def_dir, 'event_data.csv')])
    old = pd.read_csv(prev_event_path,
                      skiprows=1,
                      index_col=0, )
    temp.loc[:, 'old_temp'] = old.loc[:, 'temp'].values
    temp.loc[:, 'old_precip'] = old.loc[:, 'precip'].values
    temp.loc[:, 'change_temp'] = ~(temp.temp == temp.old_temp)
    temp.loc[:, 'change_precip'] = ~(temp.precip == temp.old_precip)
    temp.to_csv(os.path.join(event_def_dir, "event_data_with_old.csv"))
    temp.loc[:, ['month', 'change_temp', 'change_precip']].groupby('month').sum().to_csv(
        os.path.join(event_def_dir, "event_data_sum_changes.csv"))

    # make acceptable events and visualize and other stuff
    print('making acceptable event and historical frequency')

    events = get_past_event_frequency()
    data_path = os.path.join(supporting_data_dir, 'past_event_frequency.csv')
    events.to_csv(data_path)

    data_path = os.path.join(supporting_data_dir, 'visualize_event_options.csv')
    acceptable = get_acceptable_events()
    out_data = pd.DataFrame(index=pd.Index(range(1, 13), name='month'))
    for k, v in acceptable.items():
        k = k.replace('C', 'Cold').replace('A', 'Average').replace('H', 'Hot')
        k = k.replace('W', 'Wet').replace('D', 'Dry')
        out_data.loc[np.in1d(out_data.index, v), k] = True
    out_data.to_csv(data_path)

    # make event data fixed
    print('making event_data_fixed.csv')
    fix = os.path.join(root_dir, r'BS_work\fix_csv.py')
    result = subprocess.run([sys.executable, fix, event_def_path],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    # make detrendeded restriction record
    print('making detrended restriction record')
    detrend_rest = os.path.join(root_dir, r'BS_work\f_rest_detrend.py')
    rest_data = os.path.join(supporting_data_dir, 'restriction_record.csv')
    shtemps =os.path.join(root_dir, r'BS_work\SWG\SHTemps.dat')

    result = subprocess.run([sys.executable, detrend_rest, rest_data, shtemps, supporting_data_dir],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    # make restriction probabiliities
    print('making restriction probability tables')
    rest_to_cdf = os.path.join(root_dir, r'BS_work\f_rest_to_cdf.py')
    rest_record= os.path.join(supporting_data_dir,  "restriction_record_detrend.csv")
    event_data= event_def_path
    outdir = os.path.join(root_dir, r'BS_work\IID\IrrigationRestriction')
    result = subprocess.run([sys.executable, rest_to_cdf, rest_record, event_data, outdir],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    # todo make quantile tables for the new dataset!