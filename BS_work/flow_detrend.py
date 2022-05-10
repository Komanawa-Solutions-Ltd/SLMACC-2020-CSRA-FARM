"""
This file is intended to be run as a script from command line,
This file will detrended irrigation restriction data based on provided SHtemps.dat file.

This file takes 2 agruments, arg1: path to restriction record to detrend, arg2: path to SHTemps.dat file to use

detrended Irrigation restriction record will be output to a csv file located in the base folder.


Note this was produced as of 11-5-2022 to try to detrend flow data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import curve_fit
import sys
from pathlib import Path
import os
from Climate_Shocks.climate_shocks_env import event_def_path, supporting_data_dir


def SH_model_load_in(dat_filename, data_length):
    # Load in the SH wide model data
    # load in the data from file
    model_years, model_temp_av = dat_loader(dat_filename)

    # Interpolate the data to daily resolution and then fit
    int_data, int_date = Interp_run(model_temp_av, model_years, data_length)
    alt_int_date = np.linspace(0, data_length - 1, data_length)
    return int_data, alt_int_date


def Interp_run(in_data, in_years, data_num):
    # Interpolates the data to daily resolution

    # First restrict the data to the appropriate domain
    start_year = 1971
    end_year = 2020
    start_ind = in_years.index(start_year)
    end_ind = in_years.index(end_year)
    data_sample = in_data[start_ind:end_ind + 1]

    # now need to interpolate the input data to daily
    # set the interpolation points as if the mean values represent the value half way
    # through each year and then interpolate to daily temporal resolution

    old_points = np.linspace(start_year + 0.5, end_year + 0.5, num=end_year - start_year + 1)
    new_points = np.linspace(start_year + 1, end_year - 1 / 365, num=data_num)

    interpolated_data = np.interp(new_points, old_points, data_sample)
    return interpolated_data, new_points


def dat_loader(filename):
    # Loads in a dat file and exports a numpy array
    base_data = np.genfromtxt(filename, skip_header=1)
    year = []
    average = []
    for entry in base_data:
        year.append(int(entry[0]))
        average.append(np.mean(entry[1:len(entry)]))

    return year, average


def special_curve_fit(in_data, in_date, in_model):
    p0 = [0., 0., 0., 0., 0., 0., 0., 0.]
    coeff, var_matrix = curve_fit(curve_func, [in_date, in_model], in_data, p0=p0, )
    return coeff


def curve_func(x, *p):
    # Greg's equation for curve fitting
    a0, a1, a2, a3, a4, a5, a6, a7 = p
    year_len = 365.25
    t = x[1]
    d = x[0]
    return a0 + a1 * t + a2 * np.sin(2 * np.pi * d / year_len) + a3 * np.cos(2 * np.pi * d / year_len) + a4 * np.sin(
        4 * np.pi * d / year_len) + a5 * np.cos(4 * np.pi * d / year_len) + a6 * np.sin(
        6 * np.pi * d / year_len) + a7 * np.cos(6 * np.pi * d / year_len)


def Line_based_correction(in_fit, in_data, in_model):
    # make adjustments to the data based on the line
    line_vals = in_fit[0] + in_fit[1] * in_model
    end_val = line_vals[-1]
    adjust_vals = end_val - line_vals
    out_data = abs(in_data + adjust_vals)
    return out_data, adjust_vals


# below commmented out as running from python directly rather than from callable
# if len(sys.argv) != 4:
#     print("3 arguments are required, arg1: path to restriction record csv, arg2: path to  SHTemps.dat, "
#           "arg3 path to outdir")
#     sys.exit()

# First load in the F Rest data
# filename = r'C:\Users\sloth\Desktop\work\SLMACC-2020-CSRA\Climate_Shocks\supporting_data\restriction_record.csv'
filename = os.path.join(supporting_data_dir, 'restriction_record.csv')

Data = pandas.read_csv(filename)
IR = Data.get('flow')
ir_org = Data.get('flow').copy()
root_dir = os.path.dirname(os.path.dirname(__file__))
# model_temp_filename = r'C:\Users\sloth\Desktop\work\weather-gen\SWG Full Package\SHTemps.dat'
model_temp_filename = os.path.join(root_dir, r'BS_work\SWG\SHTemps.dat')

outdir = supporting_data_dir
if not os.path.exists(outdir):
    os.makedirs(outdir)

print(np.mean(IR))
Model_data, Model_dates = SH_model_load_in(model_temp_filename, len(IR))

# Only Detrend days with IR
IR_days = IR != 0
IR_A = IR[IR_days]
IR_A_dates = Model_dates[IR_days]
IR_A_model = Model_data[IR_days]

# Fit
tmp_fit_coeff = special_curve_fit(IR_A, IR_A_dates, IR_A_model)
Adj_Var_Point, adjustments = Line_based_correction(tmp_fit_coeff, IR_A, IR_A_model)

# Add Wet days back to full vector
tmp_points = np.array(IR)
tmp_points[IR_days] = Adj_Var_Point

Data['flow'] = flow = tmp_points
print(np.mean(Data['flow']))

Data.to_csv(os.path.join(outdir, "flow_record_detrend.csv"), index=False)
print(f"saved to {outdir}/flow_record_detrend.csv")
# plot to check quality of fit
line_vals = tmp_fit_coeff[0] + tmp_fit_coeff[1] * Model_data
line_fit = curve_func([Model_dates, Model_data], *tmp_fit_coeff)

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey='all')
axs[0].plot(Model_dates, IR)
axs[0].set_title('Input Data')
axs[0].set(xlabel='Time', ylabel='IR')
axs[1].plot(Model_dates, line_fit)
axs[1].set_title('Full Fit')
axs[1].set(xlabel='Time')
axs[2].plot(Model_dates, line_vals)
axs[2].set_title('Linear fit')
axs[2].set(xlabel='Time')
plt.savefig(outdir + 'Smart_fit', dpi=1000, bbox_inches='tight')

fig, (ax, ax2) = plt.subplots(2, sharex=True, figsize=(9, 3))
ax.plot(Model_dates, ir_org, label='original', c='b')
ax.plot(Model_dates, flow, label='detrend', c='r')
ax2.plot(Model_dates, ir_org - flow)
ax2.set_title('diff')
ax.legend()
plt.show()  # todo examine
