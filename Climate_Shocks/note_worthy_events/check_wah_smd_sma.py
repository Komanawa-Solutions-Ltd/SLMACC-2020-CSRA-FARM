"""
 Author: Matt Hanson
 Created: 8/01/2021 11:59 AM
 """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import glob
from Climate_Shocks.note_worthy_events.simple_soil_moisture_pet import calc_smd_sma_wah
from Climate_Shocks.note_worthy_events.explore_wah_soil_moisture import get_sim


def make_exploritory_data(number=None):
    """
    get the internal year of the wah data and calculate smd, sma, pet.  convert rain to mm/day
    :param number: number of sims to load
    :return: dates, rain, radn, tmax, tmin, rh_min, rh_max, wind_10, mslp, smd, sma, pet
    """
    paths = list(glob.glob(
        r"C:\matt_modelling_unbackedup\Z2003_SLMACC\wathome_for_matt\*.nc"
    ))
    if number is None:
        number = len(paths)
    paths = paths[0:number]
    use_shp = (365, len(paths))
    rain, radn, tmax, = np.zeros(use_shp), np.zeros(use_shp), np.zeros(use_shp)
    tmin, rh_min, rh_max, = np.zeros(use_shp), np.zeros(use_shp), np.zeros(use_shp)
    wind_10, mslp = np.zeros(use_shp), np.zeros(use_shp)
    lat, lon = -43.372, 172.333
    bad_sims = []
    for i, p in enumerate(paths):
        data = get_sim(sim_path=p, lat=lat, lon=lon, rain_mm=False)
        data = data.loc[(data.index >= '2017-07-01') & (data.index < '2018-07-01')]
        dates = data.index
        if data.shape[0] != 365:
            bad_sims.append(i)
            continue
        rain[:, i] = data['precipitation'].values
        radn[:, i] = data['sw_flux'].values
        tmax[:, i] = data['tmax'].values
        tmin[:, i] = data['tmin'].values
        rh_min[:, i] = data['rh_min'].values
        rh_max[:, i] = data['rh_max'].values
        wind_10[:, i] = data['wind_speed'].values
        mslp[:, i] = data['mslp'].values

    # correct shapes
    print(bad_sims)
    rain = np.delete(rain, bad_sims, axis=1)
    radn = np.delete(radn, bad_sims, axis=1)
    tmax = np.delete(tmax, bad_sims, axis=1)
    tmin = np.delete(tmin, bad_sims, axis=1)
    rh_min = np.delete(rh_min, bad_sims, axis=1)
    rh_max = np.delete(rh_max, bad_sims, axis=1)
    wind_10 = np.delete(wind_10, bad_sims, axis=1)
    mslp = np.delete(mslp, bad_sims, axis=1)

    smd, sma, pet = calc_smd_sma_wah(rain, radn, tmax, tmin, rh_min, rh_max, wind_10, mslp, elv=200)
    rain *= 84600
    return dates, rain, radn, tmax, tmin, rh_min, rh_max, wind_10, mslp, smd, sma, pet


def plot_exlploritory_data(number, num_to_plot=20, vars_to_plot=['rain', 'pet', 'smd', 'sma']):
    """
    quick plotting to visually check results
    :param number:
    :param num_to_plot:
    :param vars_to_plot:
    :return:
    """
    dates, rain, radn, tmax, tmin, rh_min, rh_max, wind_10, mslp, smd, sma, pet = make_exploritory_data(number)
    fig, axs = plt.subplots(nrows=len(vars_to_plot), sharex=True)
    idxs = np.random.choice(np.arange(rain.shape[1]), (num_to_plot))
    cmap = get_cmap('tab20')
    n_scens = num_to_plot
    colors = [cmap(e / n_scens) for e in range(n_scens)]
    for ax, v in zip(axs, vars_to_plot):
        ax.set_ylabel(v)
        for idx, c in zip(idxs, colors):
            ax.plot(dates, eval(v)[:, idx])


if __name__ == '__main__':
    plot_exlploritory_data(5,2)
    plt.show()
