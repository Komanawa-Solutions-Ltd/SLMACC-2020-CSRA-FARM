"""
 Author: Matt Hanson
 Created: 7/07/2020 9:27 AM

 this is a place to hold common path names to allow simple and clear naming between multiple computers

 default google file stream locations are as follows:
 Matt Hanson : M
 Zeb Etheridge: Z
#todo write up instructions on how to use (for me in 5 years)
 """
import os
from socket import gethostname
from getpass import getuser
import sys

hostname = gethostname()
user = getuser()


def mh_backed(path):
    host_nm_dict = {

        'dickie': {
            'dumon': r"M:\My Drive\MH_work_backed"},
        'Healey': {
            'Matt Hanson': r"M:\My Drive\MH_work_backed"},
        'DESKTOP-G2QSULJ': {
            'Matt Hanson': r"M:\My Drive\MH_work_backed"}
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))
    host = host_nm_dict[hostname]
    if user not in host.keys():
        raise ValueError('{} not in established users for host {}'.format(user, host))

    return os.path.join(host[user], path)


def shared_drives(path):
    host_nm_dict = {
        'wanganui': {
            'matt_dumont': '/home/matt_dumont/google_mount_point/'
        },
        'dickie': {
            'dumon': "M:/Shared drives",
            'matt_dumont': "M:/Shared drives",
        },
        'Healey': {
            'Matt Hanson': "M:/Shared drives"},
        'DESKTOP-G2QSULJ': {
            'Matt Hanson': "M:/Shared drives"}
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))
    host = host_nm_dict[hostname]
    if user not in host.keys():
        raise ValueError('{} not in established users for host {}'.format(user, host))

    return os.path.join(host[user], path)


def mh_unbacked(path):
    host_nm_dict = {
        'wanganui': {
            'matt_dumont': '/home/matt_dumont/unbacked'
        },
        'dickie': {
            'dumon': "D:/mh_unbacked",
            'matt_dumont': "D:/mh_unbacked"
        },
        'Healey': {
            'Matt Hanson': "C:/matt_modelling_unbackedup"},
        'DESKTOP-G2QSULJ': {
            'Matt Hanson': r"C:\Users\Matt Hanson\Documents\unbacked"}
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))
    host = host_nm_dict[hostname]
    if user not in host.keys():
        raise ValueError('{} not in established users for host {}'.format(user, host))

    out = os.path.join(host[user], path)
    if not os.path.exists(host[user]):
        os.makedirs(host[user])
    return out


def tempfiles(path):
    host_nm_dict = {
        'wanganui': {
            'matt_dumont': '/home/matt_dumont/temp_files'
        },
        'dickie': {
            'dumon': r"D:\temp_files"},
        'Healey': {
            'Matt Hanson': "C:/Users/Matt Hanson/Downloads/temp_python_files"},
        'DESKTOP-G2QSULJ': {
            'Matt Hanson': "C:/Users/Matt Hanson/Downloads/temp_python_files"}
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))
    host = host_nm_dict[hostname]
    if user not in host.keys():
        raise ValueError('{} not in established users for host {}'.format(user, host))

    out = os.path.join(host[user], path)

    if not os.path.exists(host[user]):
        os.makedirs(host[user])

    return out


def add_basgra_nz_path():
    host_nm_dict = {
        'wanganui': {
            'matt_dumont': '/home/matt_dumont/PycharmProjects/BASGRA_NZ_PY'
        },
        'dickie': {
            'matt_dumont': r'C:\Users\matt_dumont\PycharmProjects\BASGRA_NZ_PY',
            'dumon': r"C:\Users\dumon\python_projects\BASGRA_NZ_PY",
        },
        'Healey': {
            'Matt Hanson': 'C:/Users/Matt Hanson/python_projects/BASGRA_NZ_PY'},
        'DESKTOP-G2QSULJ': {
            'Matt Hanson': 'C:/Users/Matt Hanson/python_projects/BASGRA_NZ_PY'}
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))
    host = host_nm_dict[hostname]
    if user not in host.keys():
        raise ValueError('{} not in established users for host {}'.format(user, host))

    sys.path.append(host[user])


def get_vscn_dir():
    host_nm_dict = {
        'wanganui': {
            'matt_dumont': '/home/matt_dumont/temp_files'  # todo dummy
        },

        'dickie': {
            'dumon': r"D:\SLMMAC_SWG/SLMACC-Subset_vcsn",
            'matt_dumont': r"D:\SLMMAC_SWG/SLMACC-Subset_vcsn"
        },
        'Healey': {
            'Matt Hanson': r"D:\SLMMAC_SWG/SLMACC-Subset_vcsn"},
        'DESKTOP-G2QSULJ': {
            'Matt Hanson': r"D:\SLMMAC_SWG/SLMACC-Subset_vcsn"
        }
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))
    host = host_nm_dict[hostname]
    if user not in host.keys():
        raise ValueError('{} not in established users for host {}'.format(user, host))

    return host[user]


slmmac_dir = shared_drives('Z2003_SLMACC')
slmmac_dir_unbacked = mh_unbacked('SLMACC_2020_norm')

proj_root = os.path.dirname(__file__)

if not os.path.exists(slmmac_dir_unbacked):
    os.makedirs(slmmac_dir_unbacked)

basgra_version = '5.0.0'
