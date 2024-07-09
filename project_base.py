"""
 Author: Matt Hanson
 Created: 7/07/2020 9:27 AM
 """

from komanawa.kslcore import KslEnv
from pathlib import Path
import shutil

project_name = 'Z20002SLM_SLMACC'
slmmac_dir = KslEnv.shared_drive(project_name)
unbacked_dir = KslEnv.unbacked.joinpath(project_name)
unbacked_dir.mkdir(exist_ok=True)

proj_root = Path(__file__).parent

basgra_version = '7.0.1'


def get_vscn_dir():
    backed_dir = KslEnv.large_working.joinpath('backed/SLMMAC_2020/SLMACC-Subset_vcsn')
    local_dir = unbacked_dir.joinpath('SLMACC-Subset_vcsn')
    if local_dir.exists():
        pass
    else:
        assert backed_dir.exists(), 'backed dir does not exist: {}'.format(backed_dir)
        assert len(list(backed_dir.glob('*.nc'))) == 49, 'no vcsn files in {}'.format(backed_dir)
        print('copying vcsn files to local dir')
        shutil.copytree(backed_dir, local_dir)

    assert len(list(local_dir.glob('*.nc'))) == 49, 'no vcsn files in {}'.format(local_dir)
    return local_dir


def get_irrigation_gen_vfinal():
    backed_dir = KslEnv.large_working.joinpath('backed/SLMMAC_2020/irrigation_gen_vfinal')
    local_dir = unbacked_dir.joinpath('irrigation_gen_vfinal')
    if local_dir.exists():
        pass
    else:
        assert backed_dir.exists(), 'backed dir does not exist: {}'.format(backed_dir)
        assert len(list(backed_dir.glob('*.nc'))) == 1, 'no vcsn files in {}'.format(backed_dir)
        print('copying irrigation_gen_vfinal to local dir')
        shutil.copytree(backed_dir, local_dir)
    return local_dir


def get_gen_vfinal():
    backed_dir = KslEnv.large_working.joinpath('backed/SLMMAC_2020/gen_vfinal')
    local_dir = unbacked_dir.joinpath('gen_vfinal')
    if local_dir.exists():
        pass
    else:
        assert backed_dir.exists(), 'backed dir does not exist: {}'.format(backed_dir)
        assert len(list(backed_dir.glob('*.nc'))) == 1, 'no vcsn files in {}'.format(backed_dir)
        print('copying gen_vfinal to local dir')
        shutil.copytree(backed_dir, local_dir)
    return local_dir


def get_stocastic_weather_gen_dir():
    remote_dir = KslEnv.large_working.joinpath('backed/SLMMAC_2020/full_SWG')
    local_dir = unbacked_dir.joinpath('SWG_runs', 'full_SWG')

    if local_dir.exists():
        pass
    else:
        assert remote_dir.exists(), 'remote dir does not exist: {}'.format(remote_dir)
        assert len(list(remote_dir.glob('*.nc'))) == 104, 'wrong file number in {}'.format(remote_dir)
        print('copying stocastic_weather generator to local dir')
        shutil.copytree(remote_dir, local_dir)
    return local_dir
