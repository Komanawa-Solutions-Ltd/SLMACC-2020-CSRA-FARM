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
from Climate_Shocks.get_past_record import get_restriction_record

alternate_rest_dir = os.path.join(ksl_env.proj_root, 'Ecological_flows/v2/alternate_restrictions')

new_flows = (
    # (allocation, minimum flow)
    ()  # todo need to make these, 5, 10 20m3/s in either direction
)

new_flows = {f'a{a}-mf{m}': (a, m) for a, m in new_flows}  # todo maybe new names


def naturalise_historical_flow():
    """
    just natualise for WIL scheme
    :return:
    """
    start_year = 1999
    data = get_restriction_record()
    irrigation_start = 244  # DOY, ignoreing leap years... cause who cares
    irrigation_stop = 121  # doy, ignoreing leap years... cause who cares
    data.loc[:, 'nat'] = data.loc[:, 'flow']
    idx = (data.year >= start_year) & ((data.doy <= irrigation_stop) | (data.doy >= irrigation_start))
    data.loc[idx, 'nat'] += data.loc[idx, 'take']

    return data


def make_new_rest_record(name, nat):  # todo
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
    temp = make_new_rest_record(name, nat)
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


def get_new_flow_rest_record(name, version):
    """

    :param name: flow name (see new_flows) or 'base'
    :param version: trended v detrended
    :return:
    """

    if name == 'base':
        rest = get_restriction_record(version=version)
        return rest
    else:
        if version == 'trended':
            path = os.path.join(alternate_rest_dir, f'{name}-restriction_record.csv')
        elif version == 'detrended':
            path = os.path.join(alternate_rest_dir, f'{name}-detrend_restriction_record.csv')
        else:
            raise ValueError(f'unexpected value for version: {version}')
        dt_format = '%Y-%m-%d'
        int_keys = {
            'day': int,
            'doy': int,
            'month': int,
            'year': int,
            'f_rest': float,
            'flow': float,
            'take': float,
        }
        data = pd.read_csv(path, dtype=int_keys)
        data.loc[:, 'date'] = pd.to_datetime(data.loc[:, 'date'], format=dt_format)
        data.set_index('date', inplace=True)
        data.sort_index(inplace=True)

        return data


if __name__ == '__main__':
    naturalise_historical_flow()
    for f in new_flows:  # todo run first
        make_new_rest_data(f)
