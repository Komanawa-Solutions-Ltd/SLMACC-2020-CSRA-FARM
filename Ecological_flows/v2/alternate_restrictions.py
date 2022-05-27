"""
created matt_dumont 
on: 27/05/22
"""
import pandas as pd
import numpy as np
import os
import ksl_env
import subprocess
import sys
from Climate_Shocks.climate_shocks_env import event_def_path, supporting_data_dir
from Storylines.irrigation_mapper import get_irr_by_quantile

alternate_rest_dir = os.path.join(ksl_env.proj_root, 'Ecological_flows/v2/alternate_restrictions')

new_flows = (
    # (allocation, minimum flow)
    ()  # todo need to make these
)

new_flows = {f'a{a}-mf{m}': (a, m) for a, m in new_flows}


def naturalise_historical_flow():  # todo
    raise NotImplementedError


def make_new_rest_record(name):  # todo
    """
    convert naturalised flow record into restriction record
    :param name: name of the scenario see new_flows variable
    :return:
    """
    raise NotImplementedError


def make_new_rest_data(name):
    """
    applies the new regime to the flow, runs the detrending and exports the rest mappers
    :param name:
    :return:
    """
    os.makedirs(alternate_rest_dir, exist_ok=True)

    # naturalise* the flow record
    nat = naturalise_historical_flow()

    # make the new restriction record and save to file
    rest_data = os.path.join(alternate_rest_dir, f'{name}-restriction_record.csv')
    temp = make_new_rest_record(name)
    temp.to_csv(rest_data)

    # detrend the new restriction record
    print('detrending making detrended restriction record')
    detrend_rest = os.path.join(alternate_rest_dir, f'{name}-detrend_restriction_record.csv')
    shtemps = os.path.join(ksl_env.proj_root, r'BS_work\SWG\SHTemps.dat')

    result = subprocess.run([sys.executable, detrend_rest, rest_data, shtemps, supporting_data_dir],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))

    # make restriction mappers:
    irr_quantile_dir = os.path.join(alternate_rest_dir, f'{name}-rest_mapper')
    get_irr_by_quantile(recalc=True, outdir=irr_quantile_dir, rest_path=detrend_rest)


if __name__ == '__main__':
    for f in new_flows:  # todo run first
        make_new_rest_data(f)
