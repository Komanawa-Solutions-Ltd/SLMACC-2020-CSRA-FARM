"""
 Author: Matt Hanson
 Created: 23/02/2021 10:27 AM
 """
import pandas as pd
import numpy as np
import os
import project_base
import subprocess
import sys
import glob

from Climate_Shocks.note_worthy_events.make_norm_event_data import make_data
from Climate_Shocks.climate_shocks_env import event_def_path, supporting_data_dir
from Storylines.check_storyline import get_past_event_frequency, get_acceptable_events

if __name__ == '__main__':
    t = input('this will re-run most of the system and takes c. 15 HOURS are you sure you want to proceed '
              'with make_new_event_definition_data.py \nY/N')
    if t.lower() != 'y':
        raise ValueError('stopped to prevent override')

    re_run_SWG = False

    event_def_dir = ksl_env.slmmac_dir.joinpath(r"event_definition/norm")
    if not os.path.exists(event_def_dir):
        os.makedirs(event_def_dir)
    vcsn_version = 'detrended2'

    root_dir = os.path.dirname(os.path.dirname(__file__))

    # make detrendeded restriction record
    print('making detrended restriction record')
    detrend_rest = os.path.join(root_dir, r'BS_work\f_rest_detrend.py')
    rest_data = os.path.join(supporting_data_dir, 'restriction_record.csv')
    shtemps = os.path.join(root_dir, r'BS_work\SWG\SHTemps.dat')
    outpath = os.path.join(supporting_data_dir, "restriction_record_detrend.csv")

    result = subprocess.run([sys.executable, detrend_rest, rest_data, shtemps, outpath],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    # make event data
    print('making final event_data for vcsn: {} and event_def_dir: {}'.format(vcsn_version, event_def_dir))

    make_data(save=True,
              save_paths=[event_def_path, os.path.join(event_def_dir, 'event_data.csv')])

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

    # make restriction probabiliities
    print('making restriction probability tables')
    rest_to_cdf = os.path.join(root_dir, r'BS_work\f_rest_to_cdf.py')
    rest_record = os.path.join(supporting_data_dir, "restriction_record_detrend.csv")
    event_data = event_def_path
    outdir = os.path.join(root_dir, r'BS_work\IID\IrrigationRestriction')
    result = subprocess.run([sys.executable, rest_to_cdf, rest_record, event_data, outdir],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    from Climate_Shocks.note_worthy_events.inverse_percentile import calc_doy_per_from_historical
    from Storylines.storyline_runs.run_SWG_for_all_months import generate_all_swg, generate_SWG_output_support, \
        clean_individual
    from Storylines.irrigation_mapper import get_irr_by_quantile

    # make restriction mappers:
    get_irr_by_quantile(recalc=True)

    #  make quantile tables for the new event_data!,
    data = calc_doy_per_from_historical('detrended2')  # this should be the one used, others are for investigation
    data.to_csv(os.path.join(os.path.dirname(event_def_path), 'daily_percentiles_detrended_v2.csv'))

    from BS_work.SWG.SWG_wrapper import get_monthly_smd_mean_detrended

    get_monthly_smd_mean_detrended(False, True)

    if re_run_SWG:
        # make probality of creating an event with SWG
        prob_dir = os.path.join(ksl_env.unbacked_dir, 'SWG_runs', 'id_prob')
        generate_SWG_output_support()  # this will run one of each which makes things faster, but requires a pool of 1
        generate_all_swg(1000, False, outdir=prob_dir)
        from BS_work.SWG.check_1_month_runs import make_event_prob

        make_event_prob(prob_dir)
        # run SWG
        full_dir = os.path.join(ksl_env.unbacked_dir, 'SWG_runs', 'full_SWG')
        generate_all_swg(10000, True, full_dir)
        clean_individual(full_dir, duplicate=False)

        # run irrigation generator...
        from Climate_Shocks.Stochastic_Weather_Generator.irrigation_generator import get_irrigation_generator

        get_irrigation_generator(recalc=True)

    # run historical baseline
    from Pasture_Growth_Modelling.historical_average_baseline import get_historical_average_baseline

    get_historical_average_baseline(site='eyrewell', mode='irrigated', years=[2024], recalc=True)
    get_historical_average_baseline(site='oxford', mode='irrigated', years=[2024], recalc=True)
    get_historical_average_baseline(site='oxford', mode='dryland', years=[2024], recalc=True)

    from Storylines.storyline_building_support import make_irr_rest_for_all_events, make_blank_storyline_sheet

    make_irr_rest_for_all_events()
    make_blank_storyline_sheet()
    from Climate_Shocks.make_transition_overview import get_all_zero_prob_transitions

    try:
        get_all_zero_prob_transitions(save=True)  # this will not run without BS input
    except Exception:
        print('could not make zero transition probs')

    from Storylines.storyline_evaluation.storyline_eval_support import get_pgr_prob_baseline_stiched

    print(get_pgr_prob_baseline_stiched(1, 'eyrewell', 'irrigated'))
    print(get_pgr_prob_baseline_stiched(1, 'oxford', 'irrigated'))
    print(get_pgr_prob_baseline_stiched(1, 'oxford', 'dryland'))


    # other scripts worth re-running/ rethinking
    # Storylines/storyline_runs/run_unique_events.py # re-run unique events to see any changes
    # Storylines/storyline_building_support.py # re-consider base events
    # Storylines/base_storylines_old.py # consider base storyline and base long storyline with major changes, along with ibasal
    # Storylines/generate_random_storylines.py # it would be good to re-run random storylines
    # Storylines/storyline_runs/base_scen_long.py # if major event change, running this to support ibasal set.
    # Pasture_Growth_Modelling/basgra_parameter_sets.py # possibly re-set BASALI for major changes
