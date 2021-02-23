"""
 Author: Matt Hanson
 Created: 3/11/2020 9:07 AM
 """
import numpy as np
import pandas as pd


def calc_smd(rain, pet, h2o_cap, h2o_start, a=0.0073,
             p=1, return_drn_aet=False):
    """
    calculate the soil moisture deficit from aet, assuems that if these are arrays axis 0 is time
    :param rain: array of rain fall amounts, mm, shape = (time, *other dimensions (if needed))
    :param pet: array of pet amounts, mm, shape = (time, *other dimensions (if needed))
    :param h2o_cap: maximum soil water capacity, mm, niwa uses 150mm as a standard
    :param h2o_start: fraction of the soil water capacity to start at, fraction 0-1
    :param a: "readily available" water coefficient (d mm-1)
              default value from woodward 2010
    :param p: proportion of readilay avalible water (RAW) abe to be extracted in one day (d-1)
              default value from woodward, 2010
    :param return_drn_aet: boolean, if True return AET and drainage
    :return: (soil moisture deficit, mm) or (soil moisture deficit, mm), (drainage, mm), (aet, mm)
    """
    # make this work if float/ndarray passed
    if np.atleast_1d(pet).ndim == 1:
        array_d = 1  # 1d array or list, return 1d data
        pet = np.atleast_1d(pet[:, np.newaxis])
        rain = np.atleast_1d(rain[:, np.newaxis])
    else:
        array_d = 2  # 2 or more dimensions return without modification
    assert rain.shape == pet.shape, 'rain and PET must be same shape'
    assert h2o_start <= 1 and h2o_start >= 0, 'h2o start must be the fraction, between 0-1'

    smd = np.zeros(pet.shape, float)
    if return_drn_aet:
        drain = np.zeros(pet.shape, float)
        aet_out = np.zeros(pet.shape, float)

    iter_shp = pet.shape[1:]
    soil_mois = np.zeros((iter_shp)) + h2o_cap * h2o_start

    for i, (r, pe) in enumerate(zip(rain, pet)):
        aet = calc_aet(pe, p=p, a=a, AWHC=h2o_cap, W=soil_mois - h2o_cap)

        soil_mois = soil_mois + r - aet
        soil_mois[soil_mois < 0] = 0

        d = np.zeros((iter_shp))

        idx = soil_mois > h2o_cap
        d[idx] = soil_mois[idx] - h2o_cap
        soil_mois[idx] = h2o_cap

        smd[i] = (soil_mois - h2o_cap)

        if return_drn_aet:
            drain[i] = d
            aet_out[i] = aet

    # manage shape and return data
    if array_d == 1:
        smd = smd[:, 0]
        if return_drn_aet:
            drain = drain[:, 0]
            aet_out = aet_out[:, 0]

    if return_drn_aet:
        return smd, drain, aet_out
    else:
        return smd


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

    RAW = a * PET * (AWHC + W) * p

    AET = PET
    AET[AET > RAW] = RAW[AET > RAW]

    return AET


def calc_sma_smd_historical(rain, pet, date, h2o_cap, h2o_start, average_start_year=1981, average_stop_year=2010,
                            a=0.0073,
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

    smd, drain, aet_out = calc_smd(rain, pet, h2o_cap, h2o_start, a, p, return_drn_aet=True)

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


def calc_monthly_based_smd_sma_150mm(rain, pet, date):
    month_start = {1: -79.0, 2: -92.0, 3: -84.0, 4: -71.0, 5: -46.0, 6: -21.0, 7: -9.0, 8: -7.0, 9: -12.0, 10: -30.0,
                   11: -47.0, 12: -67.0}
    doy = pd.Series(date).dt.dayofyear.values
    month = pd.Series(date).dt.month.values
    year = pd.Series(date).dt.year.values
    day = pd.Series(date).dt.day.values

    assert date.shape == pet.shape == rain.shape, 'date, pet, rain must be same shape'

    outdata = pd.DataFrame(index=date)
    for k in ['doy', 'month', 'year', 'day', 'rain', 'pet']:
        outdata.loc[:, k] = eval(k)

    for m in range(1, 13):
        for y in range(year.max()):
            idx = (outdata.month == m) & (outdata.year == y)
            smd = calc_smd(outdata.loc[idx, 'rain'].values, outdata.loc[idx, 'pet'].values,
                           h2o_cap=150, h2o_start=(150 + month_start[m]) / 150, return_drn_aet=False)
            outdata.loc[idx, 'smd'] = smd

    return outdata


def calc_penman_pet(rad, temp, rh, wind_10=None, wind_2=None, psurf=None, mslp=None,
                    elevation=None):
    """
    calculate penman-monteith pet, works with either numeric values or with an np.ndarray.
    :param rad: radiation mJ/m2/day
    :param temp: mean temperature degrees C
    :param rh: relative humidity (percent)
    :param wind_10: 10 m wind speed (m/s) or None, one of (wind_10, wind_2) must be passed,
                    this is converted to 2m windspeed internally
    :param wind_2: 2 m wind speed (m/s) or None, one of (wind_10, wind_2) must be passed
    :param psurf: surface pressure (kpa) or None, one of psurf, mslp must be passed
    :param mslp: mean sea level pressure (kpa) or None, one of psurf, mslp must be passed
    :param elevation: elevation (m) of the point or None, needed only if mslp passed
    :return: pet (mm/day)
    """
    # check inputs
    assert (wind_10 is not None) or (wind_2 is not None), 'either wind_10 or wind_2 must not be None'
    assert (wind_10 is None) or (wind_2 is None), 'only one of wind_10 and wind_2 may not be None'
    assert (psurf is not None) or (mslp is not None), 'either psurf or mslp must not be None'
    assert (psurf is None) or (mslp is None), 'only one of psurf and mslp may not be None'

    # calc psurf if necessary
    if psurf is None:
        assert elevation is not None, 'if mslp is passed instead of psurf elevation must also be passed'
        # from https://keisan.casio.com/keisan/image/Convertpressure.pdf
        psurf = mslp * (1 - 0.0065 * elevation / (temp + 273 + 0.0065 * elevation)) ** 5.257

    # get the correct wind speed
    if wind_2 is not None:
        wind = wind_2
    elif wind_10 is not None:
        wind = wind_10 * (4.87 / (np.log(67.8 * 10 - 5.42)))
    else:
        raise ValueError('should not get here')

    # assure it is the correct size etc.
    err_mess = ('non-matching shapes, [rad, temp, rh, wind_10 | wind_2=None, psurf| mslp] must be the same shape, '
                'problem with temp and {}')
    for v in ['rad', 'wind', 'rh', 'psurf']:
        assert np.atleast_1d(temp).shape == np.atleast_1d(eval(v)).shape, err_mess.format(v)

    h_vap = 02.501 - 0.00236 * temp  # latent heat of vaporisation

    tmp = 4098 * (0.6108 * np.e ** ((17.27 * temp) / (temp + 237.3)))
    delt = tmp / (temp + 237.3) ** 2  # gradient of the vapour pressure curve Based on equation 13 in Allen et al (1998)

    soil = 0  # soil heat flux (set to zero by niwa)
    y = (1.1013 * psurf) / (0.622 * h_vap * 1000)  # piesometric constant
    es = 0.61094 * np.e ** (17.625 * temp / (243.04 + temp))
    ed = rh * es / 100

    pet = ((h_vap ** -1 * delt * (rad - soil) + y * (900 / (temp + 273)) * wind * (es - ed)) /
           (delt + y * (1 + 0.34 * wind)))
    return pet

def calc_smd_sma_wah_monthly(rain, radn, tmax, tmin, rh_min, rh_max, wind_10, mslp, elv):
    """
    calculate soil moisture deficit, soil moisture anomaly, and pet for weather at home data.  this is a convenience
    function for Bodeker Scientific.  the expected inputs which are nd arrays are expected to be 2d arrays of
    shape (365, num of sims) the goal soil moisture anomaly is calculated against the mean(axis=1) of the soil moisture
    deficit array.  The units should be in the same format as weather at home. assumes no input data contains nan values
    SMD assumes a starting soil moisture of 75mm and a water holding capacity of 150mm
    :param rain: precipitation (kg m-2 s-1), np.ndarray
    :param radn: radiation (W m-2), np.ndarray
    :param tmax: maximum temperature (k), np.ndarray
    :param tmin: minimum temperature (k), np.ndarray
    :param rh_min: maximum relative humidity (%), np.ndarray
    :param rh_max: minimum relative humidity (%), np.ndarray
    :param wind_10: 10m wind speed (m/s), np.ndarray
    :param mslp: mean sea level pressure (Pa), np.ndarray
    :param elv: elevation at site (m), float
    :return: smd(mm), sma(mm), pet(mm/day)
    """
    raise NotImplementedError # todo make this function for Bryn to run on his W@H data, pull from below

def calc_smd_sma_wah_depreciated(rain, radn, tmax, tmin, rh_min, rh_max, wind_10, mslp, elv):
    """
    calculate soil moisture deficit, soil moisture anomaly, and pet for weather at home data.  this is a convenience
    function for Bodeker Scientific.  the expected inputs which are nd arrays are expected to be 2d arrays of
    shape (365, num of sims) the goal soil moisture anomaly is calculated against the mean(axis=1) of the soil moisture
    deficit array.  The units should be in the same format as weather at home. assumes no input data contains nan values
    SMD assumes a starting soil moisture of 75mm and a water holding capacity of 150mm
    :param rain: precipitation (kg m-2 s-1), np.ndarray
    :param radn: radiation (W m-2), np.ndarray
    :param tmax: maximum temperature (k), np.ndarray
    :param tmin: minimum temperature (k), np.ndarray
    :param rh_min: maximum relative humidity (%), np.ndarray
    :param rh_max: minimum relative humidity (%), np.ndarray
    :param wind_10: 10m wind speed (m/s), np.ndarray
    :param mslp: mean sea level pressure (Pa), np.ndarray
    :param elv: elevation at site (m), float
    :return: smd(mm), sma(mm), pet(mm/day)
    """
    raise ValueError('depreciated')
    # check inputs
    expected_shape = rain.shape
    assert (expected_shape[0] == 366 or
            expected_shape[0] == 365), 'axis 0 must be days and it is expected to be a full year (365 or 366 days)'
    assert len(expected_shape) == 2, 'expected 2d array (day of year, simulation number)'
    for k in ['radn', 'rain', 'tmax', 'tmin', 'rh_min', 'rh_max', 'wind_10', 'mslp']:
        assert eval(k).shape == expected_shape, '{} does not match rain shape'.format(k)
        assert np.isfinite(
            eval(k)).all(), 'nan values passed in {}, please remove otherwise they will impact the sma'.format(k)

    # make mean values and convert units
    temp = (tmax + tmin) / 2 - 273.15  # to C
    rh = (rh_min + rh_max) / 2
    rain = rain * 86400  # kg/m2/s to mm/day
    radn = radn * 86400 * 1e-6  # from w/m2 to mj/m2/day
    mslp = mslp / 1000  # Pa to kpa

    # run SMD/SMA
    pet = calc_penman_pet(rad=radn, temp=temp, rh=rh, wind_10=wind_10, wind_2=None, psurf=None, mslp=mslp,
                          elevation=elv)
    smd = calc_smd(rain=rain, pet=pet, h2o_cap=150, h2o_start=0.5, a=0.0073,
                   p=1, return_drn_aet=False)
    sma = smd - smd.mean(axis=1)[:, np.newaxis]
    return smd, sma, pet


def rough_testing_of_pet():
    # rough testing, looks good enough, and matches external
    import matplotlib.pyplot as plt
    from Climate_Shocks.note_worthy_events.fao import fao56_penman_monteith, delta_svp, svp_from_t, psy_const, \
        avp_from_rhmean

    data = pd.read_csv(r"M:\Shared drives\Z2003_SLMACC\event_definition\hamilton_weather.csv").set_index(
        ['year', 'doy'])
    temp = pd.read_csv(r"M:\Shared drives\Z2003_SLMACC\event_definition\penman.csv").set_index(['year', 'doy'])
    data.loc[temp.index, 'penman'] = temp.loc[:, 'penman']
    for i in data.index:
        t = (data.loc[i, 'tmin'] + data.loc[i, 'tmax']) / 2 + 273
        svp = svp_from_t(t - 273)
        data.loc[i, 'penman_calc_ext'] = fao56_penman_monteith(
            net_rad=data.loc[i, 'radn'],
            t=t,
            ws=data.loc[i, 'wind'],
            svp=svp,
            avp=avp_from_rhmean(svp, svp, data.loc[i, 'rh']),
            delta_svp=delta_svp(t - 273),
            psy=psy_const(data.loc[i, 'pmsl'] / 10),
        )
        data.loc[i, 'iter_penman'] = calc_penman_pet(rad=data.loc[i, 'radn'],
                                                     temp=t - 273,
                                                     rh=data.loc[i, 'rh'],
                                                     wind_2=data.loc[i, 'wind'],
                                                     mslp=data.loc[i, 'pmsl'] / 10,
                                                     elevation=45)
    # my pet
    data.loc[:, 'penman_calc'] = calc_penman_pet(rad=data['radn'].values,
                                                 temp=(data['tmin'].values + data['tmax'].values) / 2,
                                                 rh=data['rh'].values,
                                                 wind_2=data['wind'].values,
                                                 mslp=data['pmsl'].values / 10,
                                                 elevation=45
                                                 )

    data = data.rolling(10).mean()
    data.plot(y=['penman', 'penman_calc', 'penman_calc_ext', 'iter_penman'])
    # plt.scatter(data['pet'], data['peyman_pet'])
    plt.show()
    pass


detrended_start_month = {
    1: -50.0, 2: -52.0, 3: -41.0, 4: -36.0, 5: -22.0, 6: -11.0, 7: -7.0, 8: -8.0,
    9: -14.0, 10: -30.0, 11: -41.0, 12: -49.0}
# calculated from historical SMD from detrended2 on first day of each month with a 10 day centered rolling window mean


def calc_smd_monthly(rain, pet, dates,
                     month_start=detrended_start_month,
                     h2o_cap=150,
                     a=0.0073,
                     p=1, return_drn_aet=False):
    """
    calculate the soil moisture deficit from aet, assuems that if these are arrays axis 0 is time
    sets the start of each month to the delta of the month start value and the rain/pet on that day.
    :param rain: array of rain fall amounts, mm, shape = (time, *other dimensions (if needed))
    :param pet: array of pet amounts, mm, shape = (time, *other dimensions (if needed))
    :param dates: array of datetime objects,  shape = (time,)
    :param month_start: the SMD value to start each month with
    :param h2o_cap: maximum soil water capacity, mm, niwa uses 150mm as a standard
    :param a: "readily available" water coefficient (d mm-1)
              default value from woodward 2010
    :param p: proportion of readilay avalible water (RAW) abe to be extracted in one day (d-1)
              default value from woodward, 2010
    :param return_drn_aet: boolean, if True return AET and drainage
    :return: (soil moisture deficit, mm) or (soil moisture deficit, mm), (drainage, mm), (aet, mm)
    """

    assert isinstance(month_start, dict)
    assert set(month_start.keys()) == set(range(1, 13))
    dates = pd.Series(np.atleast_1d(dates))
    months = np.array([d.month for d in dates])
    days = np.array([d.day for d in dates])
    # make this work if float/ndarray passed
    if np.atleast_1d(pet).ndim == 1:
        array_d = 1  # 1d array or list, return 1d data
        pet = np.atleast_1d(pet)[:, np.newaxis]
        rain = np.atleast_1d(rain)[:, np.newaxis]
    else:
        array_d = 2  # 2 or more dimensions return without modification
    assert rain.shape == pet.shape, 'rain, dates and PET must be same shape'
    assert dates.shape == pet.shape[0:1]
    smd = np.zeros(pet.shape, float) * np.nan
    if return_drn_aet:
        drain = np.zeros(pet.shape, float) * np.nan
        aet_out = np.zeros(pet.shape, float) * np.nan

    iter_shp = pet.shape[1:]
    soil_mois = np.zeros((iter_shp)) + month_start[months[0]] + h2o_cap

    for i, (r, pe, m, d) in enumerate(zip(rain, pet, months, days)):
        if d == 1:  # set the soil moisture on the first day of the month
            soil_mois[:] = month_start[m] + h2o_cap

        aet = calc_aet(pe, p=p, a=a, AWHC=h2o_cap, W=soil_mois - h2o_cap)

        soil_mois = soil_mois + r - aet
        soil_mois[soil_mois < 0] = 0

        d = np.zeros((iter_shp))

        idx = soil_mois > h2o_cap
        d[idx] = soil_mois[idx] - h2o_cap
        soil_mois[idx] = h2o_cap

        smd[i] = (soil_mois - h2o_cap)

        if return_drn_aet:
            drain[i] = d
            aet_out[i] = aet

    # manage shape and return data
    if array_d == 1:
        smd = smd[:, 0]
        if return_drn_aet:
            drain = drain[:, 0]
            aet_out = aet_out[:, 0]

    if return_drn_aet:
        return smd, drain, aet_out
    else:
        return smd


def test_calc_smd():
    # todo write testing regime in the future. I'm confident it is working correctly
    # float
    # 1d
    # 2d
    # 3d
    raise NotImplementedError


def test_penman_pet():
    # todo write a testing regime in the future I am confidnet that it is working appropriately
    raise NotImplementedError


if __name__ == '__main__':
    # rough_testing_of_pet()
    from Climate_Shocks.get_past_record import get_vcsn_record

    vcsn = get_vcsn_record('detrended2')
    t = calc_smd_monthly(rain=vcsn.rain, pet=vcsn.pet, dates=vcsn.index)
    vcsn.loc[:, 'smd_monthly'] = t
    t = vcsn.loc[:, ['doy', 'smd_monthly']].groupby('doy').mean().to_dict()
    vcsn.loc[:, 'sma_monthly'] = vcsn.loc[:, 'smd_monthly'] - vcsn.loc[:, 'doy'].replace(t['smd_monthly'])

    temp = calc_sma_smd_historical(vcsn.rain, vcsn.pet, vcsn.index, h2o_cap=150, h2o_start=1)
    vcsn.loc[:, 'smd'] = temp.loc[:, 'smd'].values

    vcsn.loc[:, 'sma'] = temp.loc[:, 'sma'].values
    vcsn.loc[:, 'rolling_smd'] = vcsn.loc[:, 'smd'].rolling(10, center=True).mean()
    vcsn.loc[:, 'day'] = vcsn.index.day
    t = vcsn.loc[~((vcsn.month == 2) & (vcsn.day == 29))].groupby('doy').mean()
    t.loc[:, 'day'] = t.day.astype(int)

    temp = vcsn.groupby(['year', 'month']).mean().reset_index().groupby('month').describe()
    temp.to_csv(r"C:\Users\dumon\Downloads\compare_smd-sma.csv")
