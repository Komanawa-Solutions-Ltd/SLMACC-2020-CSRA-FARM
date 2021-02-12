"""
 Author: Matt Hanson
 Created: 12/02/2021 2:28 PM
 """

def read_swg_data(paths, exsite):
    """
    read teh data and put it in the correct format for BASGRA
    :param paths: list of paths to indidivual netcdf files
    :param exsite: boolean, if True then is the extra site for oxford, if False it is the normal site, eyrewell
    :return:
    """
    outdata = []
    for p in paths:
        #todo read the data in, and pull out what's needed
        #todo start here
        raise NotImplementedError