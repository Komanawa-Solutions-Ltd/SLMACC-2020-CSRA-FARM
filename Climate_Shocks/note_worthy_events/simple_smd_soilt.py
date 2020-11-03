"""
 Author: Matt Hanson
 Created: 3/11/2020 9:07 AM
 """
import numpy as np
import pandas as pd


def calc_soil_temp():
    raise NotImplementedError  # todo hopefully pull from VCSN


# todo I may also need to run soil moisture anomaly (https://niwa.co.nz/climate/nz-drought-monitor/droughtindicatormaps/soil-moisture-deficit-smd#:~:text=SMD%20is%20calculated%20based%20on,can%20use)%20of%20150%20mm. )
def calc_smd(rain, pet, h2o_cap, h2o_start, a=0.0073,
             p=1):
    """
    calculate the soil moisture deficit from aet,
    :param rain: array of rain fall amounts, mm
    :param pet: array of pet amounts, mm
    :param h2o_cap: maximum soil water capacity, mm, niwa uses 150mm as a standard
    :param h2o_start: fraction of the soil water capacity to start at, fraction 0-1
    :param a: "readily available" water coefficient (d mm-1)
              default value from woodward 2010
    :param p: proportion of readilay avalible water (RAW) abe to be extracted in one day (d-1)
              default value from woodward, 2010
    :return: (soil moisture deficit, mm), (drainage, mm), (aet, mm)
    """
    rain = np.atleast_1d(rain)
    pet = np.atleast_1d(pet)
    assert rain.shape == pet.shape, 'rain and PET must be same length'
    assert h2o_start <= 1 and h2o_start >= 0, 'h2o start must be the fraction, between 0-1'

    smd = np.zeros(pet.shape, float)
    drain = np.zeros(pet.shape, float)
    aet_out = np.zeros(pet.shape, float)

    soil_mois = h2o_cap * h2o_start

    for i, (r, pe) in enumerate(zip(rain, pet)):
        aet = calc_aet(pe, p=p, a=a, AWHC=h2o_cap, W=soil_mois - h2o_cap)
        soil_mois = max(0, soil_mois + r - aet)
        d = 0
        if soil_mois > h2o_cap:
            d = soil_mois - h2o_cap
            soil_mois = h2o_cap

        drain[i] = d
        smd[i] = (soil_mois - h2o_cap)
        aet_out[i] = aet
    return smd, drain, aet_out


def calc_aet(PET, AWHC, W, p=1, a=0.0073):
    """
    calculate AET for new zealand pasture from
    # from woodward 2010, https://www.tandfonline.com/doi/pdf/10.1080/00288233.2001.9513464
    :param PET: potential evapotranspriation (mm/day)
    :param p: proportion of readilay avalible water (RAW) abe to be extracted in one day (d-1)
              default value from woodward 2010
    :param a: "readily available" water coefficient (d mm-1)
              default value from woodward 2010
    :param AWHC: available water holding capacity of soil to rooting depth (mm)
    :param W: soil water deficit (usually negative) (mm)
    :return:
    """

    RAW = a * PET * (AWHC + W)

    AET = min(PET, p * RAW)

    return AET


def calc_sma_smd(rain, pet, date, h2o_cap, h2o_start, average_start_year=1981, average_stop_year=2010, a=0.0073,
                 p=1):
    """
    calculate the soil moisture deficit from aet,
    :param rain: array of precip amounts, mm
    :param pet: array of pet amounts, mm
    :param date: the dates for the pet/precip data
    :param h2o_cap: maximum soil water capacity, mm, niwa uses 150mm as a standard
    :param h2o_start: fraction of the soil water capacity to start at, fraction 0-1
    :param average_start_year: start date for the averaging period, inclusive
    :param average_stop_year: end date for the averaging period, inclusive
    :param a: "readily available" water coefficient (d mm-1)
              default value from woodward 2010
    :param p: proportion of readilay avalible water (RAW) abe to be extracted in one day (d-1)
              default value from woodward, 2010
    :return: (soil moisture deficit, mm), (drainage, mm), (aet, mm)
    """
    date = np.atleast_1d(date)
    doy = pd.Series(date).dt.dayofyear

    pet = np.atleast_1d(pet)
    rain = np.atleast_1d(rain)

    assert date.shape == pet.shape == rain.shape, 'date, pet, rain must be same shape'

    smd, drain, aet_out = calc_smd(rain, pet, h2o_cap, h2o_start, a, p)

    outdata = pd.DataFrame(data={'date': date, 'doy': doy, 'pet': pet, 'rain': rain, 'smd': smd, 'drain': drain,
                                 'aet_out': aet_out},
                           )

    # calculate mean smd for doy

    idx = (outdata.date.dt.year >= average_start_year) & (outdata.date.dt.year <= average_stop_year)
    temp = outdata.loc[idx, ['doy', 'smd']]
    average_smd = temp.groupby(doy).mean().set_index('doy')

    outdata.loc[:, 'mean_doy_smd'] = outdata.loc[:, 'doy']
    outdata.replace({'mean_doy_smd': average_smd.loc[:, 'smd'].to_dict()}, inplace=True)

    outdata.loc[:, 'sma'] = outdata.loc[:, 'smd'] - outdata.loc[:, 'mean_doy_smd']

    return outdata
