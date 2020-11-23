"""
 Author: Matt Hanson
 Created: 7/07/2020 9:27 AM

 this is a place to hold common path names to allow simple and clear naming between multiple computers

 default google file stream locations are as follows:
 Matt Hanson : M
 Zeb Etheridge: Z

 """
import os
from socket import gethostname
from getpass import getuser

hostname = gethostname()
user = getuser()

def mh_backed(path):
    host_nm_dict = {
        'dickie': r"M:\My Drive\MH_work_backed",
        'Healey': r"M:\My Drive\MH_work_backed",
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))

    return os.path.join(host_nm_dict[hostname], path)


def shared_drives(path):
    user_dict = {
        'Matt Hanson': 'M:/',
        'dumon': 'M:/'

    }

    return os.path.join(user_dict[user], "Shared drives", path)


def mh_unbacked(path):
    host_nm_dict = {
        'dickie': "D:/mh_unbacked",
        'Healey': "C:/matt_modelling_unbackedup",
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))

    out = os.path.join(host_nm_dict[hostname], path)
    if not os.path.exists(host_nm_dict[hostname]):
        os.makedirs(host_nm_dict[hostname])
    return out


def tempfiles(path):
    host_nm_dict = {
        'dickie': r"D:\temp_files",
        'Healey': "C:/Users/Matt Hanson/Downloads/temp_python_files",
    }

    if hostname not in host_nm_dict.keys():
        raise ValueError('{} not in established hostnames: {}'.format(hostname, host_nm_dict.keys()))

    out = os.path.join(host_nm_dict[hostname], path)

    if not os.path.exists(host_nm_dict[hostname]):
        os.makedirs(host_nm_dict[hostname])

    return out
