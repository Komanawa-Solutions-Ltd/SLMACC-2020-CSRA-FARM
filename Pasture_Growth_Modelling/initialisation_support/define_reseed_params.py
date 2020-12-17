"""
 Author: Matt Hanson
 Created: 18/12/2020 9:30 AM
 """

from Pasture_Growth_Modelling.initialisation_support.inital_long_term_runs import run_past_basgra_irrigated, \
    run_past_basgra_dryland, plot_multiple_results
import pandas as pd
import numpy as np

if __name__ == '__main__':
    reseed=True
    irr_ox = run_past_basgra_irrigated(site='oxford',reseed=reseed)
    eyrewell = run_past_basgra_irrigated(reseed=reseed)
    dry_ox = run_past_basgra_dryland(site='oxford', reseed=reseed)
    plot_multiple_results({'irr_ox':irr_ox, 'irr_eyre':eyrewell, 'dry_ox':dry_ox},
                          out_vars=['BASAL'], rolling=30, main_kwargs={'alpha': 0.2}, label_main=False,
                          label_rolling=True, show=True)
    t = dry_ox
    t.loc[t.doy == 152].BASAL.describe(percentiles=np.arange(5, 55, 5) / 100)