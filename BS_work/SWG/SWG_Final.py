# Fully self generative mono-site stochastic weather generator
# Author: Alex Schuddeboom

#todo think about wheter to use yaml or function call for this....
import numpy as np
import netCDF4 as nc
import os
from scipy.optimize import curve_fit
import random
from scipy import stats
import pandas
import math
import yaml
import warnings
import sys

# load in required parameters

warnings.filterwarnings('ignore')
Param_file = sys.argv[1]
# Param_file='D:\Data\Bodeker\SWG_Params.yaml'

if not os.path.exists(Param_file):
    print('Cannot find Yaml file. Please check that the yaml file given as an argument exists.')
    sys.exit()

with open(Param_file, 'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    Param_dict = yaml.load(file, Loader=yaml.FullLoader)

# Extract values for the parameters
lat_val = Param_dict['lat_val']
lon_val = Param_dict['lon_val']
month_scale_factor = Param_dict['month_scale_factor']
number_of_simulations = Param_dict['number_of_simulations']
base_directory = Param_dict['base_directory']
SH_model_temp_file = Param_dict['SH_model_temp_file']
VCSN_Class_file = Param_dict['VCSN_Class_file']
simulation_savename = Param_dict['simulation_savename']
story_line_filename = Param_dict['story_line_filename']
extra_flag = Param_dict['Extra_station_flag']
netcdf_flag = Param_dict['netcdf_save_flag']
Extra_site_lat = Param_dict['Extra_site_lat']
Extra_site_lon = Param_dict['Extra_site_lon']
Extra_sim_savename = Param_dict['Extra_sim_savename']

if not extra_flag:
    Extra_site_lat = []
    Extra_site_lon = []

# Check some basic error conditions
if not os.path.exists(SH_model_temp_file):
    print('Invalid SH model file. Please change in the Yaml file.')
    sys.exit()

if not os.path.exists(VCSN_Class_file):
    print('Invalid VCSN Classification file. Please change in the Yaml file.')
    sys.exit()

if not os.path.exists(story_line_filename):
    print('Invalid storyline file. Please change in the Yaml file.')
    sys.exit()

if not os.path.exists(base_directory + 'SLMACC-Subset_vcsn/'):
    print(
        'Invalid base_directory location. This should be the folder that contains the folder SLMACC-Subset_vcsn. Please change in the Yaml file.')
    sys.exit()


def data_loader(baselevel_dir):
    # Identifies relevant files, calls the loaders and manages the output
    SLMACC_dir = baselevel_dir + 'SLMACC-Subset_vcsn/'
    SLMACC_files = os.listdir(SLMACC_dir)
    SLMACC_files.sort()
    SLMACC_data = []
    for file in SLMACC_files:
        if file[-16:] == '_for-Komanawa.nc' and not (file.__contains__('2020')):
            full_name = SLMACC_dir + file
            year_data = nc.Dataset(full_name)
            SLMACC_data.append(year_data)
    return SLMACC_data


def var_stripper(input_data, var_name):
    # given the name of a variable from the netcdf data, extract all instances of that variable
    lat = input_data[0]['lat'][:]
    lon = input_data[0]['lon'][:]
    out_data = []
    out_date = []
    for tmp_data in input_data:
        out_data.append(tmp_data[var_name][:])
        out_date.append(tmp_data['date'][:])

    return lat, lon, out_date, out_data


def lat_lon_picker(lat_data, lon_data, lat_point, lon_point):
    # finds the closest bin to a given lat lon point
    tmp_lat_vec = abs(np.array(lat_data) - lat_point)
    tmp_lon_vec = abs(np.array(lon_data) - lon_point)

    lat_ind_val = np.argmin(tmp_lat_vec)
    lon_ind_val = np.argmin(tmp_lon_vec)
    return lat_ind_val, lon_ind_val


def point_stripper(lat_ind, lon_ind, In_data, In_date):
    # given the lat-lon indices reduces the input data to one point
    out_data = []
    for year in In_data:
        for day_data in year:
            tmp_out = day_data[lat_ind][lon_ind]
            out_data.append(tmp_out)

    out_date = []
    for year in In_date:
        for day_data in year:
            tmp_out = day_data
            out_date.append(tmp_out)

    return out_data, out_date


def save_dir_gen(basic_directory, lat_ind, lon_ind):
    # Given the relevant information returns the directory to save to.
    # If the identified directory doesnt exist, creates it

    first_directory = basic_directory + 'Output' + '/'
    save_directory = basic_directory + 'Output' + '/' + str(lat_ind) + '_' + str(lon_ind) + '/'
    if not (os.path.isdir(first_directory)):
        os.mkdir(first_directory)
    if not (os.path.isdir(save_directory)):
        os.mkdir(save_directory)
    return save_directory


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


def base_data_detrender(base_directory, lat_point, lon_point, model_temp_filename):
    # Loads in and detrends the data require to run the SWG

    # base directory is where data is loaded from and will be used to generate save location
    # variable name is the load name of variable
    # lat and lon point describe the location you want to examine
    # mon_filter describes the temporal limitation: 0 for all year, 1 for jan, 2 for feb....
    # reg_filter describes the regime filtering: 0 for no filter, other nums as required

    # This first section loads and pre processes that data

    variable_list = ['pr', 'tasmax', 'tasmin', 'rsds', 'evspsblpot']

    # Load in the appropriate VCSN data
    Full_SLMACC_data = data_loader(base_directory)
    # Extract the required variables
    Var_data = []
    for var in variable_list:
        [Lat, Lon, Date, tmp_Var_data] = var_stripper(Full_SLMACC_data, var)
        Var_data.append(tmp_Var_data)
    # Isolate that variable to a single point in the VCSN data
    Var_Points = []
    point_flag = True
    while point_flag:
        lat_ind, lon_ind = lat_lon_picker(Lat, Lon, lat_point, lon_point)
        for array in Var_data:
            tmp_Var_point, Var_Date = point_stripper(lat_ind, lon_ind, array, Date)
            Var_Points.append(tmp_Var_point)

        if not (np.isnan(tmp_Var_point[0])):
            point_flag = False
        else:
            print('No appropriately close station data, please use a different lat/lon')
            print('Problem lat:' + str(lat_point))
            print('Problem lon:' + str(lon_point))
            sys.exit()

    # Determine the location where output will be saved.
    save_dir = save_dir_gen(base_directory, lat_ind, lon_ind)

    # With the preliminary loading done we can now move onto removing the trend

    # First need to load and handle the SH Model data
    Model_data, Model_dates = SH_model_load_in(model_temp_filename, len(tmp_Var_point))
    fit_coeff = []
    Adj_Var_Points = []
    # Now detrend the data. Precipitation must be handled separately so 0s can be removed
    for i in range(0, len(variable_list)):
        if variable_list[i] == 'pr':
            # Only Detrend wet days
            tmp_pr = np.array(Var_Points[i])
            wet_days = tmp_pr != 0
            PR_A = tmp_pr[wet_days]
            PR_A_dates = Model_dates[wet_days]
            PR_A_model = Model_data[wet_days]
            tmp_fit_coeff = special_curve_fit(PR_A, PR_A_dates, PR_A_model)
            fit_coeff.append(tmp_fit_coeff)
            Adj_Var_Point, adjustments = Line_based_correction(tmp_fit_coeff, PR_A, PR_A_model)
            # Add Wet days back to full vector
            tmp_points = np.array(Var_Points[i])
            tmp_points[wet_days] = Adj_Var_Point
            Adj_Var_Points.append(tmp_points)
        else:
            tmp_fit_coeff = special_curve_fit(Var_Points[i], Model_dates, Model_data)
            fit_coeff.append(tmp_fit_coeff)
            Adj_Var_Point, adjustments = Line_based_correction(tmp_fit_coeff, Var_Points[i], Model_data)
            Adj_Var_Points.append(Adj_Var_Point)
    return variable_list, Adj_Var_Points, Var_Date, save_dir


def streak_count(in_dry_days):
    # finds the streaks of dry and wet days within the data

    out_ind_streak_vals = []
    out_streak_lens = []
    out_streak_lens_flag = []

    start_flag = in_dry_days[0]
    count = 0
    for day_data in in_dry_days:
        if day_data == start_flag:
            out_ind_streak_vals.append(count)
            count += 1
        else:
            out_ind_streak_vals.append(0)
            out_streak_lens.append(count)
            count = 1
            if start_flag:
                out_streak_lens_flag.append('Dry')
            else:
                out_streak_lens_flag.append('Wet')
            start_flag = day_data

    return out_streak_lens, out_streak_lens_flag, out_ind_streak_vals


def streak_dependant_transition_probabilities(in_streaks, day_flags):
    # Calculates the probability of a transition occurring for each given
    # streak length

    tmp_streaks = np.array(in_streaks)
    new_streaks = tmp_streaks[day_flags]
    # alt_streaks = np.array(new_streaks[0:-1])
    max_streak = np.max(new_streaks)
    inds = np.array(range(0, len(new_streaks)))

    Total_count = []
    end_flag = False
    for i in range(0, max_streak):
        tmp_inds = inds[new_streaks == i]
        if len(tmp_inds) == 0:
            end_flag = True
        if not end_flag:
            Total_count.append(len(tmp_inds))

    return Total_count  # , Transition_count


def probability_plot(in_wet, in_dry):
    # Plots the probability of transition given state
    prob_wet = []
    for i in range(0, len(in_wet) - 1):
        prob_val_w = in_wet[i + 1] / in_wet[i]
        if prob_val_w <= 1:
            prob_wet.append(prob_val_w)
        else:
            prob_wet.append(1)

    prob_dry = []
    for i in range(0, len(in_dry) - 1):
        prob_val_d = in_dry[i + 1] / in_dry[i]
        if prob_val_d <= 1:
            prob_dry.append(prob_val_d)
        else:
            prob_dry.append(1)

    return prob_wet, prob_dry


def exp_curve_func(z, *p):
    # Exponential fitting function
    a, b, c, d = p
    return a * np.exp(b * z + d) + c


def log_curve_func(R, *p):
    # Logarithmic fitting function
    a, b, c, d = p
    return (1 / b) * (np.log((R - c) / a) - d)


def Precipitation_function_fit(PR_amounts):
    # Fits the precipitation amount CDF

    hist, bin_edge = np.histogram(PR_amounts, bins=100)
    n_hist = hist / hist.sum()
    cdf = np.cumsum(n_hist)
    bin_centres = (np.array(bin_edge[0:len(bin_edge) - 1]) + np.array(bin_edge[1:len(bin_edge)])) / 2

    p0 = [0., 0., 0., 0.]
    coeff, var_matrix = curve_fit(exp_curve_func, bin_centres, cdf, p0=p0)

    if log_curve_func(0.00, *coeff) < 0:
        new_coeff = np.array([0., 0., 0., 0.])

        new_coeff[0] = coeff[0]
        new_coeff[1] = coeff[1]
        new_coeff[2] = coeff[2]
        new_coeff[3] = np.log(-1 * coeff[2] / coeff[0])

        coeff = new_coeff

    return coeff


def PR_streak_calc(In_precipitation_data):
    # Pre filtering PR calculations
    dry_days = In_precipitation_data == 0

    streak_lengths, streak_type, streak_values = streak_count(dry_days)
    streak_values = np.array(streak_values)
    return streak_values


def SWG_Reg_Precipitation(In_precipitation_data, streak_values):
    # Calculates all of the precipitation variables needed to run the SWG
    np_precip = np.array(In_precipitation_data)
    dry_days = np_precip == 0

    Wet_Streak_size = streak_dependant_transition_probabilities(streak_values, dry_days == False)
    Dry_Streak_size = streak_dependant_transition_probabilities(streak_values, dry_days)

    Wet_probability, Dry_probability = probability_plot(Wet_Streak_size, Dry_Streak_size)

    Precipitation_fit = Precipitation_function_fit(np_precip[dry_days == False])
    return Precipitation_fit, dry_days, Wet_probability, Dry_probability


def Data_filter(in_data, in_date, in_streaks, regime_filename, mon_filter=0, reg_filter=0):
    # Filters the data down to just appropriate subset

    # mon_filter describes the temporal limitation: 0 for all year, 1 for jan, 2 for feb....
    # reg_filter describes the regime filtering: 0 for no filter, other nums as required

    # First do the monthly filtering

    mon_data = []

    if mon_filter == 0:
        mon_str = '_all'
        mon_data = in_data
        mon_date = in_date
        mon_streaks = in_streaks
        last_day = [False] * len(in_data[0])
    else:
        mon_str = '_' + str(mon_filter)
        for array in in_data:
            tmp_mon_data, mon_date, last_day = Month_Filtering(array, in_date, mon_filter)
            mon_data.append(tmp_mon_data)
        mon_streaks, _, _ = Month_Filtering(in_streaks, in_date, mon_filter)

    out_data = []
    # Filter data to requested regimes. Will need to load in appropriate data
    if reg_filter == 0:
        reg_str = '_all'
        out_data = mon_data
        out_date = mon_date
        out_precip_data = mon_data[0]
        out_streaks = mon_streaks
        new_last_day = last_day
    else:
        # some test data for the regime filtering
        reg_str = '_' + str(reg_filter)
        # load in local regime identifying file
        regime_data, reg_year, reg_months = regime_data_loader(regime_filename)

        # need to determine which days are apart of which regimes
        regime_list = regime_filter(mon_date, regime_data)
        for array in mon_data:
            tmp_out_data, out_date = Regime_Filtering(array, mon_date, regime_list, reg_filter)
            out_data.append(tmp_out_data)

        new_last_day, out_date = Regime_Filtering(last_day, mon_date, regime_list, reg_filter)
        out_streaks, out_date = Regime_Filtering(mon_streaks, mon_date, regime_list, reg_filter)

        # Need to create the precipitation filtered data. First find all of the
        # precipitation relevant regimes and then filter data
        mod_reg_value = np.mod(reg_filter, 3)
        if mod_reg_value == 0:
            precip_filters = [3, 6, 9]
        elif mod_reg_value == 1:
            precip_filters = [1, 4, 7]
        elif mod_reg_value == 2:
            precip_filters = [2, 5, 8]

        out_precip_data = []
        for mon_val in precip_filters:
            tmp_out_data, test_out_date = Regime_Filtering(mon_data[0], mon_date, regime_list, mon_val)
            out_precip_data = out_precip_data + tmp_out_data

    combined_str = mon_str + reg_str

    return out_data, out_date, out_precip_data, out_streaks, combined_str, new_last_day


def regime_data_loader(file_name):
    # loads in the local regime identifying files
    base_data = pandas.read_csv(file_name, comment="#")
    temp_cl = base_data.get('temp_class')
    precip_cl = base_data.get('precip_class')
    year = base_data.get('year')
    months = base_data.get('month')

    try:
        len(temp_cl)
        len(precip_cl)
    except:
        print(
            'Error reading in the historical classifications file. Please ensure that it has been converted to the '
            'correct format with fix_csv.py.')
        sys.exit()

    regime_list = []
    for i in range(0, len(temp_cl)):
        pr_val = precip_cl[i]
        tmp_val = temp_cl[i]

        viable_PR = ['W', 'AP', 'A', 'D']
        viable_T = ['C', 'AT', 'A', 'H']

        if not (viable_PR.__contains__(pr_val)) or not (viable_T.__contains__(tmp_val)):
            print(
                'Historical classifications file uses an unexpected climate bin value - ' + pr_val + ' or ' + tmp_val + '.')
            print('Please check that fix_csv.py has been used on the historic data.')
            sys.exit()

        if pr_val == 'W':
            p_ind = 1
        elif pr_val == 'AP' or pr_val == 'A':
            p_ind = 2
        else:
            p_ind = 3

        if tmp_val == 'C':
            t_ind = 0
        elif tmp_val == 'AT' or tmp_val == 'A':
            t_ind = 1
        else:
            t_ind = 2

        regime_list.append(3 * t_ind + p_ind)
    return regime_list, year, months


def regime_filter(in_dates, in_regimes):
    # given a particular date set, filters the regime data down to a relevant data vector

    # iterate through all of the in dates and identify corresponding regime
    new_regime_list = []
    for i in range(0, len(in_dates)):
        tmp_date = str(in_dates[i])
        tmp_year = int(tmp_date[0:4])
        tmp_mon = int(tmp_date[4:6])

        year_ind = tmp_year - 1972
        list_ind = year_ind * 12 + tmp_mon - 1
        new_regime_list.append(in_regimes[list_ind])

    return new_regime_list


def Month_Filtering(in_data, in_dates, filter_num):
    # Takes the point data and filters it to requested month

    if filter_num > 9:
        check_str = str(filter_num)
    else:
        check_str = '0' + str(filter_num)

    out_data = []
    out_dates = []
    last_day_in_mon = []
    for date in in_dates:
        mon_val = str(date)[4:6]
        if check_str == mon_val:
            out_data.append(in_data[in_dates.index(date)])
            out_dates.append(date)
            if date == in_dates[-1]:
                last_day_in_mon.append(True)
            elif str(in_dates[in_dates.index(date) + 1])[4:6] != mon_val:
                last_day_in_mon.append(True)
            else:
                last_day_in_mon.append(False)

    return out_data, out_dates, last_day_in_mon


def Regime_Filtering(in_data, in_dates, filter_list, filter_num):
    # Takes the point data and filters it to requested bucket

    out_data = []
    out_dates = []
    count = 0
    for value in filter_list:
        if filter_num == value:
            out_data.append(in_data[count])
            out_dates.append(in_dates[count])
        count += 1

    return out_data, out_dates


def A_B_Builder(Tmax, Tmin, R, EV, end_mon_vec):
    # generates the A and B matrix from the filtered data

    # Calculate Z scores
    z_data = [(Tmax - np.mean(Tmax)) / np.std(Tmax),
              (Tmin - np.mean(Tmin)) / np.std(Tmin),
              (R - np.mean(R)) / np.std(R),
              (EV - np.mean(EV)) / np.std(EV)]

    # Need to handle the data so that it doesn't mess up on month to month transitions
    # First generate a vector with all of the month ends dropped
    drop_end = []
    drop_start = []
    count = 0
    for data in z_data:
        raw_inds = np.array(range(0, len(end_mon_vec)))
        inds = raw_inds[np.array(end_mon_vec) == False]
        first_inds = inds + 1
        if sum(np.array(end_mon_vec) == True) == 0:
            drop_end.append(data[0:-1])
            drop_start.append(data[1:])
        else:
            drop_end.append(data[inds])
            drop_start.append(data[first_inds])
        count += 1

    # Calculate the correlation co-efficients

    m0 = np.zeros((4, 4))
    m1 = np.zeros((4, 4))
    count = 0
    for item in z_data:
        count2 = 0
        for item2 in z_data:
            m0[count, count2] = stats.pearsonr(item, item2)[0]
            m1[count, count2] = stats.pearsonr(drop_start[count], drop_end[count2])[0]
            count2 += 1
        count += 1

    # Now that we have the m0 and m1 need to calculate the A and B matrix

    A = np.matmul(m1, np.linalg.inv(m0))
    S = m0 - np.matmul(A, m1.transpose())
    B = np.linalg.cholesky(S)

    return A, B, z_data


def fake_init(in_data, in_streaks, in_transformed_data):
    # An initialization that can used prior to a proper initializer being built

    # should return the simulation relevant list [pr_o, pr_a, pr_streak, tmax, tmin, rs, ev]

    # Initializes the sequence by picking a random day from the dataset
    R_day = random.randrange(len(in_data[0]))
    # variables to be initialized
    init_vals = []

    # handle precipitation
    pr_value = in_data[0][R_day]
    if pr_value == 0 or pr_value == 0.0:
        init_vals.append('D')
    else:
        init_vals.append('W')
    init_vals.append(pr_value)
    init_vals.append(in_streaks[R_day])

    for array in in_transformed_data:
        init_vals.append(array[R_day])

    return init_vals


def cdf_builder(in_data, in_dry_days):
    # Makes CDFs for the data
    # Need one for wet and one for dry
    W_Full_CDF = []
    W_Bin_list = []
    D_Full_CDF = []
    D_Bin_list = []

    bin_num = 100

    for i in range(0, len(in_data)):
        tmp_data = np.array(in_data[i])
        dry_data = tmp_data[in_dry_days]
        wet_data = tmp_data[in_dry_days == False]

        hist, bin_edge = np.histogram(dry_data, bins=bin_num)
        n_hist = hist / hist.sum()

        bin_centres = (np.array(bin_edge[0:len(bin_edge) - 1]) + np.array(bin_edge[1:len(bin_edge)])) / 2
        interp_bins = np.linspace(bin_centres[0], bin_centres[-1], num=len(bin_centres) * 10)
        interp_cdf = np.interp(interp_bins, bin_centres, np.cumsum(n_hist))

        D_Full_CDF.append(interp_cdf)
        D_Bin_list.append(interp_bins)

        hist_2, bin_edge_2 = np.histogram(wet_data, bins=bin_num)
        n_hist_2 = hist_2 / hist_2.sum()

        bin_centres_2 = (np.array(bin_edge_2[0:len(bin_edge) - 1]) + np.array(bin_edge_2[1:len(bin_edge)])) / 2
        interp_bins_2 = np.linspace(bin_centres_2[0], bin_centres_2[-1], num=len(bin_centres_2) * 10)
        interp_cdf_2 = np.interp(interp_bins_2, bin_centres_2, np.cumsum(n_hist_2))

        W_Full_CDF.append(interp_cdf_2)
        W_Bin_list.append(interp_bins_2)

    return W_Full_CDF, W_Bin_list, D_Full_CDF, D_Bin_list


def PR_CDF_gen(in_data):
    # Given the precipitation state data returns the CDF and Bins
    bin_num = 1000
    tmp_data = np.array(in_data)
    wet_days = tmp_data != 0
    wet_data = tmp_data[wet_days]

    hist, bin_edge = np.histogram(wet_data, bins=bin_num)
    n_hist = hist / hist.sum()
    Cdf = np.cumsum(n_hist)

    bin_centres = (np.array(bin_edge[0:len(bin_edge) - 1]) + np.array(bin_edge[1:len(bin_edge)])) / 2
    # interp_bins = np.linspace(bin_centres[0], bin_centres[-1], num=len(bin_centres) * 10)
    # interp_cdf = np.interp(interp_bins, bin_centres, np.cumsum(n_hist))

    return Cdf, bin_centres


def data_load_or_make(in_lat_val, in_lon_val, in_Month_filter, in_Regime_filter, in_base_directory, model_filename,
                      in_regime_filename):
    # loads or makes the data as required

    # needed presets

    lat_vec = np.linspace(-44.025, -41.475, 52)
    lon_vec = np.linspace(168.975, 174.025, 102)
    save_lat_ind, save_lon_ind = lat_lon_picker(lat_vec, lon_vec, lat_val, lon_val)

    # Need to determine if data can be loaded or if it must be generated
    if in_Month_filter == 0:
        Month_tag = 'all'
    else:
        Month_tag = str(in_Month_filter)

    if in_Regime_filter == 0:
        Regime_tag = 'all'
    else:
        Regime_tag = str(in_Regime_filter)

    station_directory = in_base_directory + 'Output' + '/' + str(save_lat_ind) + '_' + str(save_lon_ind) + '/'
    load_filename = station_directory + 'SWG_Run_Data_' + Month_tag + '_' + Regime_tag + '.npz'
    if not (os.path.isdir(station_directory)):
        gen_data = True
    elif not (os.path.exists(load_filename)):
        gen_data = True
    else:
        gen_data = False

    if gen_data:
        print('Processed data does not exist, generating now.....')
        # Detrend the variables
        Variable_names, Adjusted_Variables, Variable_times, Save_directory = base_data_detrender(in_base_directory,
                                                                                                 in_lat_val, in_lon_val,
                                                                                                 model_filename)
        PR_Streaks = PR_streak_calc(Adjusted_Variables[Variable_names.index('pr')])
        # Filter the data according to the requested months and regimes
        Filter_data, Filter_date, Filter_precip_data, Filter_PR_streaks, Filter_string, Filter_month_ends = Data_filter(
            Adjusted_Variables,
            Variable_times,
            PR_Streaks,
            in_regime_filename,
            in_Month_filter,
            in_Regime_filter)

        if len(Filter_data[0]) == 0:
            print('Matt you idiot! You requested a storyline that does not exist historically (month number ' + str(
                in_Month_filter) + ' regime number ' + str(in_Regime_filter) + ')')
            sys.exit()

        # Calculate the Required precipitation variables
        PR_fit, Dry_days, PR_Wet_Prob, PR_Dry_Prob = SWG_Reg_Precipitation(Filter_data[Variable_names.index('pr')],
                                                                           Filter_PR_streaks)

        PR_CDF, PR_CDF_bins = PR_CDF_gen(Filter_precip_data)

        # If any variables need to be transformed this should be the space

        # Generate the CDFS for the required variables
        Wet_CDFs, Wet_CDF_bins, Dry_CDFs, Dry_CDF_bins = cdf_builder([Filter_data[Variable_names.index('tasmax')],
                                                                      Filter_data[Variable_names.index('tasmin')],
                                                                      Filter_data[Variable_names.index('rsds')],
                                                                      Filter_data[Variable_names.index('evspsblpot')]],
                                                                     Dry_days)

        # Calculate the A and B matrix
        A_matrix, B_matrix, Z_Values = A_B_Builder(Filter_data[Variable_names.index('tasmax')],
                                                   Filter_data[Variable_names.index('tasmin')],
                                                   Filter_data[Variable_names.index('rsds')],
                                                   Filter_data[Variable_names.index('evspsblpot')],
                                                   Filter_month_ends)

        np.savez(Save_directory + 'SWG_Run_Data' + Filter_string, Variable_names=Variable_names, Z_Values=Z_Values,
                 Filter_data=Filter_data, Wet_CDFs=Wet_CDFs, Wet_CDF_bins=Wet_CDF_bins, Dry_CDFs=Dry_CDFs,
                 Dry_CDF_bins=Dry_CDF_bins,
                 Filter_date=Filter_date, PR_Dry_Prob=PR_Dry_Prob, PR_Wet_Prob=PR_Wet_Prob, PR_Streaks=PR_Streaks,
                 Dry_days=Dry_days, PR_fit=PR_fit, A_matrix=A_matrix, B_matrix=B_matrix, PR_CDF=PR_CDF,
                 PR_CDF_bins=PR_CDF_bins)
    else:
        Full_data = np.load(load_filename)
        Variable_names = Full_data['Variable_names']
        Z_Values = Full_data['Z_Values']
        Filter_data = Full_data['Filter_data']
        Filter_date = Full_data['Filter_date']
        PR_Dry_Prob = Full_data['PR_Dry_Prob']
        PR_Wet_Prob = Full_data['PR_Wet_Prob']
        PR_Streaks = Full_data['PR_Streaks']
        Dry_days = Full_data['Dry_days']
        PR_fit = Full_data['PR_fit']
        A_matrix = Full_data['A_matrix']
        B_matrix = Full_data['B_matrix']
        Wet_CDFs = Full_data['Wet_CDFs']
        Wet_CDF_bins = Full_data['Wet_CDF_bins']
        Dry_CDFs = Full_data['Dry_CDFs']
        Dry_CDF_bins = Full_data['Dry_CDF_bins']
        PR_CDF = Full_data['PR_CDF']
        PR_CDF_bins = Full_data['PR_CDF_bins']

    return Variable_names, Z_Values, Filter_data, Wet_CDFs, Wet_CDF_bins, Dry_CDFs, Dry_CDF_bins, Filter_date, PR_Dry_Prob, PR_Wet_Prob, PR_Streaks, Dry_days, PR_fit, A_matrix, B_matrix, PR_CDF, PR_CDF_bins


def Does_it_change_state(active_streak, Probability):
    # Determines if it will rain based on if it rained yesterday
    if len(Probability) == 0:
        persistence_probability = 0.05
    elif active_streak < len(Probability):
        persistence_probability = Probability[active_streak]
    else:
        persistence_probability = np.mean(Probability)

    R_value = random.random()
    if R_value > persistence_probability:
        change = True
    else:
        change = False

    return change


def Random_precipitation_distribution(R, *p):
    # Given the precipitation distribution coefficients and a random number
    # returns a precipitation value
    a, b, c, d = p

    return (1 / b) * (np.log(((R - c) / a)) - d)


def Precipitation_Simulator(Fit_values):
    # Given that it is confirmed to be raining makes an estimation of precipitation amount
    R_value = random.random() * Fit_values[2]  # adjust the R value to prevent log(0) error
    simulated_rain_value = Random_precipitation_distribution(R_value, *Fit_values)
    return simulated_rain_value


def zptile(z_score):
    # calculate percentile from z-score
    return .5 * (math.erf(z_score / 2 ** .5) + 1)


def percent2cdf(percentile, cdf, bin_points):
    # given a percentile and cdf return a value
    diff_array = np.abs(cdf - percentile)
    min_ind = np.argmin(diff_array)
    cdf_point = cdf[min_ind]

    standard_flag = True
    if percentile > cdf_point and min_ind != len(cdf) - 1:
        new_ind = min_ind + 1
        cdf_point_2 = cdf[new_ind]
    elif percentile < cdf_point and min_ind != 0:
        new_ind = min_ind - 1
        cdf_point_2 = cdf[new_ind]
    else:
        cdf_point_2 = cdf_point
        standard_flag = False

    # now need to roughly interpolate between two points
    if standard_flag and cdf_point != cdf_point_2:
        w1 = np.abs(cdf_point - percentile) / np.abs(cdf_point - cdf_point_2)
        w2 = np.abs(cdf_point_2 - percentile) / np.abs(cdf_point - cdf_point_2)
        out_val = w1 * bin_points[min_ind] + w2 * bin_points[new_ind]
    else:
        out_val = bin_points[min_ind]
    return out_val


def SWG(in_simulation_length, in_Initial_data, in_PR_Wet_Prob, in_PR_Dry_Prob, in_PR_fit, in_A_matrix, in_B_matrix,
        in_Wet_CDFs, in_Wet_CDF_bins, in_Dry_CDFs, in_Dry_CDF_bins, in_PR_CDF, in_PR_CDF_bins):
    # The SWG Function
    L_D_state = []
    L_D_PR_A = []
    L_D_PR_streak = []
    L_D_Tmax = []
    L_D_Tmin = []
    L_D_RSDS = []
    L_D_EV = []

    for day in range(0, in_simulation_length):
        # O_data is the values from yesterday
        # D_vars are the variables for the simulated day

        # Handle initialization
        if day == 0:
            O_data = in_Initial_data

        # First need to simulation the precipitation occurrence
        if O_data[0] == 'W':
            # First determine if the state changes or not
            change_state = Does_it_change_state(O_data[2], in_PR_Wet_Prob)
            # Then based on he state change simulate precipitation amount if required
            if change_state:
                D_PR_state = 'D'
                D_PR_A = 0
                D_PR_streak = 0
            else:
                D_PR_state = 'W'
                D_PR_streak = O_data[2] + 1
                # D_PR_A = Precipitation_Simulator(in_PR_fit)

                R_value = random.random()
                D_PR_A = percent2cdf(R_value, in_PR_CDF, in_PR_CDF_bins)
        else:
            change_state = Does_it_change_state(O_data[2], in_PR_Dry_Prob)
            if not change_state:
                D_PR_state = 'D'
                D_PR_A = 0
                D_PR_streak = O_data[2] + 1
            else:
                D_PR_state = 'W'
                D_PR_streak = 0
                # D_PR_A = Precipitation_Simulator(in_PR_fit)

                R_value = random.random()
                D_PR_A = percent2cdf(R_value, in_PR_CDF, in_PR_CDF_bins)
        # Now need to simulate all of the remaining variables

        # First need to calculate z(t)
        z_vec_Y = [O_data[3], O_data[4], O_data[5], O_data[6]]

        epsilon_vec = np.random.multivariate_normal(np.array([0, 0, 0, 0]), np.identity(4))
        Z_vec = np.matmul(in_A_matrix, z_vec_Y) + np.matmul(in_B_matrix, epsilon_vec)

        # Now this z value can be transformed into the physical variable based on PR state

        # First Calculate the expected percentile from z value
        Percentile_Tmax = zptile(Z_vec[0])
        Percentile_Tmin = zptile(Z_vec[1])
        Percentile_RSDS = zptile(Z_vec[2])
        Percentile_EV = zptile(Z_vec[3])

        # Next use this to calculate the value with a cdf fit
        if D_PR_state == 'W':
            D_Tmax = percent2cdf(Percentile_Tmax, in_Wet_CDFs[0], in_Wet_CDF_bins[0])
            D_Tmin = percent2cdf(Percentile_Tmin, in_Wet_CDFs[1], in_Wet_CDF_bins[1])
            D_Rad = percent2cdf(Percentile_RSDS, in_Wet_CDFs[2], in_Wet_CDF_bins[2])
            D_EV = percent2cdf(Percentile_EV, in_Wet_CDFs[3], in_Wet_CDF_bins[3])
        else:
            D_Tmax = percent2cdf(Percentile_Tmax, in_Dry_CDFs[0], in_Dry_CDF_bins[0])
            D_Tmin = percent2cdf(Percentile_Tmin, in_Dry_CDFs[1], in_Dry_CDF_bins[1])
            D_Rad = percent2cdf(Percentile_RSDS, in_Dry_CDFs[2], in_Dry_CDF_bins[2])
            D_EV = percent2cdf(Percentile_EV, in_Dry_CDFs[3], in_Dry_CDF_bins[3])

        # Handle outputs and set up loop to be refreshed
        L_D_state.append(D_PR_state)
        L_D_PR_A.append(D_PR_A)
        L_D_PR_streak.append(D_PR_streak)
        if D_Tmax >= D_Tmin:
            L_D_Tmax.append(D_Tmax)
            L_D_Tmin.append(D_Tmin)
        else:
            L_D_Tmax.append(D_Tmin)
            L_D_Tmin.append(D_Tmax)
        L_D_RSDS.append(D_Rad)
        L_D_EV.append(D_EV)

        # Update O_data for next day of simulation
        N_O_data = [D_PR_state, D_PR_A, D_PR_streak, Z_vec[0], Z_vec[1], Z_vec[2], Z_vec[3]]
        O_data = N_O_data
    return L_D_state, L_D_PR_A, L_D_PR_streak, L_D_Tmax, L_D_Tmin, L_D_RSDS, L_D_EV, O_data


def Storyline_loader(file_name):
    # Given a storyline file, loads and exports relevant arrays
    base_data = pandas.read_csv(file_name, comment="#")
    temp_cl = base_data.get('temp_class')
    if temp_cl is None:
        temp_cl = base_data.get(' temp_class')
    precip_cl = base_data.get('precip_class')
    if precip_cl is None:
        precip_cl = base_data.get(' precip_class')

    if type(base_data['month'][0]) == str:
        look_up = {'all': 0, 'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5,
                   'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        months = base_data['month'].apply(lambda x: look_up[x])
    else:
        months = base_data.get('month')

    regime_list = []
    for i in range(0, len(temp_cl)):
        pr_val = precip_cl[i].strip()
        tmp_val = temp_cl[i].strip()

        if pr_val == 'W':
            p_ind = 1
        elif pr_val == 'A' or pr_val == 'AP':
            p_ind = 2
        else:
            p_ind = 3

        if tmp_val == 'C':
            t_ind = 0
        elif tmp_val == 'A' or tmp_val == 'AT':
            t_ind = 1
        else:
            t_ind = 2

        if pr_val == 'all' and tmp_val == 'all':
            regime_list.append(0)
        else:
            regime_list.append(3 * t_ind + p_ind)

    return regime_list, months


def Extra_site_loader(in_lat_val, in_lon_val, in_Month_filter, in_Regime_filter, in_base_directory, base_data,
                      in_model_filename, in_regime_filename):
    # generate or make data for a given extra point

    # needed presets

    lat_vec = np.linspace(-44.025, -41.475, 52)
    lon_vec = np.linspace(168.975, 174.025, 102)
    save_lat_ind, save_lon_ind = lat_lon_picker(lat_vec, lon_vec, in_lat_val, in_lon_val)

    # Need to determine if data can be loaded or if it must be generated
    if in_Month_filter == 0:
        Month_tag = 'all'
    else:
        Month_tag = str(in_Month_filter)

    if in_Regime_filter == 0:
        Regime_tag = 'all'
    else:
        Regime_tag = str(in_Regime_filter)

    station_directory = in_base_directory + 'Output' + '/' + str(save_lat_ind) + '_' + str(save_lon_ind) + '/'
    load_filename = station_directory + 'SWG_Extra_Data_' + Month_tag + '_' + Regime_tag + '.npz'
    if not (os.path.isdir(station_directory)):
        gen_data = True
    elif not (os.path.exists(load_filename)):
        gen_data = True
    else:
        gen_data = False

    if gen_data:
        # If data doesnt exist make it
        print('Processed data for an extra site does not exist, generating now.....')

        # Detrend the variables
        in_Variable_names, Adjusted_Variables, Variable_times, Save_directory = base_data_detrender(in_base_directory,
                                                                                                    in_lat_val,
                                                                                                    in_lon_val,
                                                                                                    in_model_filename)

        in_PR_Streaks = PR_streak_calc(Adjusted_Variables[in_Variable_names.index('pr')])

        # Filter the data according to the requested months and regimes
        in_Filter_data, in_Filter_date, Filter_precip_data, Filter_PR_streaks, Filter_string, Filter_month_ends = Data_filter(
            Adjusted_Variables, Variable_times, in_PR_Streaks, in_regime_filename, in_Month_filter, in_Regime_filter)

        PR_cdf_flag, E_PR_Cdf, E_PR_bin, E_PR_VEC = Site_data_comp_PR(in_Filter_data, base_data)

        E_CDFs, E_Bins = Extra_site_CDF_gen(in_Filter_data, base_data)

        np.savez(Save_directory + 'SWG_Extra_Data' + Filter_string, Variable_names=in_Variable_names,
                 Filter_data=in_Filter_data, Filter_date=in_Filter_date, PR_cdf_flag=PR_cdf_flag, E_PR_Cdf=E_PR_Cdf,
                 E_PR_bin=E_PR_bin, E_PR_VEC=E_PR_VEC, E_CDFs=E_CDFs, E_Bins=E_Bins)
    else:
        # if data exists load it!
        Full_data = np.load(load_filename)
        in_Variable_names = Full_data['Variable_names']
        in_Filter_data = Full_data['Filter_data']
        PR_cdf_flag = Full_data['PR_cdf_flag']
        E_PR_Cdf = Full_data['E_PR_Cdf']
        E_PR_bin = Full_data['E_PR_bin']
        E_PR_VEC = Full_data['E_PR_VEC']
        E_CDFs = Full_data['E_CDFs']
        E_Bins = Full_data['E_Bins']

    return in_Variable_names, in_Filter_data, PR_cdf_flag, E_PR_Cdf, E_PR_bin, E_PR_VEC, E_CDFs, E_Bins


def Site_data_comp_PR(in_site_data, in_base_data):
    # Given site data and base data returns joint rain probabilities, rain transition CDFs and other Var transition cdfs

    # First rain probabilities
    PR_VEC = np.zeros(4)
    base_precipitation = in_base_data[0]
    base_dry = base_precipitation == 0

    site_precipitation = np.array(in_site_data[0])
    site_dry = site_precipitation[base_dry]
    site_wet = site_precipitation[base_dry == False]
    PR_VEC[0] = np.sum(site_wet != 0) / len(site_wet)  # rain at extra given rain at base
    PR_VEC[1] = np.sum(site_wet == 0) / len(site_wet)  # dry at extra given rain at base
    PR_VEC[2] = np.sum(site_dry != 0) / len(site_dry)  # rain at extra given dry at base
    PR_VEC[3] = np.sum(site_dry == 0) / len(site_dry)  # dry at extra given dry at base

    # Now precipitation correlation
    index = np.logical_and(base_dry == False, site_precipitation != 0)
    if sum(index) != 0:
        PR_data_flag = True
        PR_Transition_values = site_precipitation[index] - base_precipitation[index]

        bin_num = 100
        hist, bin_edge = np.histogram(PR_Transition_values, bins=bin_num)
        n_hist = hist / hist.sum()
        PR_Cdf = np.cumsum(n_hist)
        PR_bin_centres = (np.array(bin_edge[0:len(bin_edge) - 1]) + np.array(bin_edge[1:len(bin_edge)])) / 2
    else:
        PR_data_flag = False
        PR_A_data = site_precipitation[site_precipitation != 0]

        bin_num = 1000
        hist, bin_edge = np.histogram(PR_A_data, bins=bin_num)
        n_hist = hist / hist.sum()
        PR_Cdf = np.cumsum(n_hist)
        PR_bin_centres = (np.array(bin_edge[0:len(bin_edge) - 1]) + np.array(bin_edge[1:len(bin_edge)])) / 2

    return PR_data_flag, PR_Cdf, PR_bin_centres, PR_VEC


def Extra_site_CDF_gen(in_site_data, in_base_data):
    # generate the transition cdfs for the remaining variables
    CDFs = []
    Bins = []
    for ex_ind in range(1, len(in_site_data)):
        tmp_site_data = np.array(in_site_data[ex_ind])
        tmp_base_data = np.array(in_base_data[ex_ind])

        transition_data = tmp_site_data - tmp_base_data

        bin_num = 100
        hist, bin_edge = np.histogram(transition_data, bins=bin_num)
        n_hist = hist / hist.sum()
        tmp_Cdf = np.cumsum(n_hist)
        tmp_bin_centres = (np.array(bin_edge[0:len(bin_edge) - 1]) + np.array(bin_edge[1:len(bin_edge)])) / 2
        CDFs.append(tmp_Cdf)
        Bins.append(tmp_bin_centres)
    return CDFs, Bins


def sim_flatten(in_data):
    # flatten data to 1d
    out_data = []
    for mon in in_data:
        out_data = out_data + list(mon)
    return out_data


def netcdf_saver(in_sim_PR_A, in_sim_Tmax, in_sim_Tmin, in_sim_RSDS, in_sim_EV, in_sim_mon_vals,
                 in_savename, in_lat_val, in_lon_val):
    # save netcdf output

    # First flatten to 1 d
    PR_A_flat = sim_flatten(in_sim_PR_A)
    Tmax_flat = sim_flatten(in_sim_Tmax)
    Tmin_flat = sim_flatten(in_sim_Tmin)
    RSDS_flat = sim_flatten(in_sim_RSDS)
    EV_flat = sim_flatten(in_sim_EV)
    mon_flat = sim_flatten(in_sim_mon_vals)
    day_list = list(range(0, len(PR_A_flat)))

    fn = in_savename + '.nc'
    ds = nc.Dataset(fn, 'w', format='NETCDF4')

    ds.lat = in_lat_val
    ds.lon = in_lon_val

    day_dim = ds.createDimension('day', None)

    day = ds.createVariable('day', 'f4', ('day',))
    day.long_name = 'Day of the simulation'
    day[:] = day_list

    PR_A = ds.createVariable('PR_A', np.float64, ('day'))
    PR_A[:] = np.array(PR_A_flat)
    PR_A.long_name = 'Precipitation'
    PR_A.units = "kg m-2 s-1"
    Tmax = ds.createVariable('Tmax', np.float64, ('day'))
    Tmax[:] = np.array(Tmax_flat)
    Tmax.long_name = 'Daily Maximum Near-Surface Air Temperature'
    Tmax.units = "K"
    Tmin = ds.createVariable('Tmin', np.float64, ('day'))
    Tmin[:] = np.array(Tmin_flat)
    Tmin.long_name = 'Daily Minimum Near-Surface Air Temperature'
    Tmin.units = "K"
    RSDS = ds.createVariable('RSDS', np.float64, ('day'))
    RSDS[:] = np.array(RSDS_flat)
    RSDS.long_name = 'Surface Downwelling Shortwave Radiation'
    RSDS.units = "W m-2"
    PEV = ds.createVariable('PEV', np.float64, ('day'))
    PEV[:] = np.array(EV_flat)
    PEV.long_name = 'Potential Evapotranspiration'
    PEV.units = "kg m-2"
    Month = ds.createVariable('Month', np.float64, ('day'))
    Month[:] = np.array(mon_flat)
    Month.long_name = 'Month being simulated'
    ds.close()


# load in storyline to be simulated
Storyline_regimes, Storyline_months = Storyline_loader(story_line_filename)
month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * month_scale_factor

# Initialize the dataset

for sim_ind in range(0, number_of_simulations):
    # Iterate through each simulation
    sim_state = []
    sim_PR_A = []
    sim_PR_streak = []
    sim_Tmax = []
    sim_Tmin = []
    sim_RSDS = []
    sim_EV = []

    sim_mon_vals = []
    for i_mon in range(0, len(Storyline_regimes)):
        # iterate through every month in a given simulation

        # load/make data
        Variable_names, Z_Values, Filter_data, Wet_CDFs, Wet_CDF_bins, Dry_CDFs, Dry_CDF_bins, Filter_date, PR_Dry_Prob, PR_Wet_Prob, PR_Streaks, Dry_days, PR_fit, A_matrix, B_matrix, PR_CDF, PR_CDF_bins = data_load_or_make(
            lat_val, lon_val, Storyline_months[i_mon], Storyline_regimes[i_mon], base_directory, SH_model_temp_file,
            VCSN_Class_file)

        if i_mon == 0:
            Initial_data = fake_init(Filter_data, PR_Streaks, Z_Values)
        else:
            Initial_data = sim_O_data

        simulation_length = month_lengths[Storyline_months[i_mon] - 1]
        # run SWG
        [tmp_sim_state, tmp_sim_PR_A, tmp_sim_PR_streak, tmp_sim_Tmax, tmp_sim_Tmin, tmp_sim_RSDS, tmp_sim_EV,
         sim_O_data] = SWG(simulation_length, Initial_data, PR_Wet_Prob, PR_Dry_Prob, PR_fit,
                           A_matrix, B_matrix, Wet_CDFs, Wet_CDF_bins, Dry_CDFs, Dry_CDF_bins, PR_CDF, PR_CDF_bins)

        sim_state.append(tmp_sim_state)
        sim_PR_A.append(tmp_sim_PR_A)
        sim_PR_streak.append(tmp_sim_PR_streak)
        sim_Tmax.append(tmp_sim_Tmax)
        sim_Tmin.append(tmp_sim_Tmin)
        sim_RSDS.append(tmp_sim_RSDS)
        sim_EV.append(tmp_sim_EV)
        sim_mon_vals.append([Storyline_months[i_mon]] * simulation_length)

    # Save output
    print('Simulation ' + str(sim_ind + 1) + ' out of ' + str(number_of_simulations) + ' complete for main station')
    tmp_simulation_savename = simulation_savename + '_S' + str(sim_ind)
    if not netcdf_flag or extra_flag:
        np.savez(tmp_simulation_savename, PR_State=sim_state, PR_Amount=sim_PR_A, PR_Streak=sim_PR_streak,
             Tmax=sim_Tmax, Tmin=sim_Tmin, RSDS=sim_RSDS, EV=sim_EV)
    if netcdf_flag:
        netcdf_saver(sim_PR_A, sim_Tmax, sim_Tmin, sim_RSDS, sim_EV, sim_mon_vals, tmp_simulation_savename, lat_val,
                     lon_val)

# Need to handle the extra sites

for i_ex in range(0, len(Extra_site_lat)):
    # iterate through each site
    for sim_ind in range(0, number_of_simulations):
        # iterate through each simulation
        sim_E_state = []
        sim_E_PR_A = []
        sim_E_Tmax = []
        sim_E_Tmin = []
        sim_E_RSDS = []
        sim_E_EV = []
        sim_E_mon_vals = []

        simulation_load_name = simulation_savename + '_S' + str(sim_ind) + '.npz'
        simulation_data = np.load(simulation_load_name, allow_pickle=True)

        sim_state = simulation_data['PR_State']
        sim_PR_A = simulation_data['PR_Amount']
        sim_Tmax = simulation_data['Tmax']
        sim_Tmin = simulation_data['Tmin']
        sim_RSDS = simulation_data['RSDS']
        sim_EV = simulation_data['EV']
        for i_mon in range(0, len(Storyline_regimes)):
            # Load in/make the required data
            Variable_names, Z_Values, Filter_data, Wet_CDFs, Wet_CDF_bins, Dry_CDFs, Dry_CDF_bins, Filter_date, PR_Dry_Prob, PR_Wet_Prob, PR_Streaks, Dry_days, PR_fit, A_matrix, B_matrix, PR_CDF, PR_CDF_bins = data_load_or_make(
                lat_val, lon_val, Storyline_months[i_mon], Storyline_regimes[i_mon], base_directory, SH_model_temp_file,
                VCSN_Class_file)

            Extra_site_names, Extra_site_data, E_P_F, E_P_CDF, E_P_bin, E_P_vec, E_CDF, E_bins = Extra_site_loader(
                Extra_site_lat[i_ex], Extra_site_lon[i_ex], Storyline_months[i_mon], Storyline_regimes[i_mon],
                base_directory, Filter_data, SH_model_temp_file, VCSN_Class_file)

            Full_e_PR_state = []
            Full_e_PR_A = []
            Full_e_tmax = []
            Full_e_tmin = []
            Full_e_rsds = []
            Full_e_ev = []

            tmp_sim_state = sim_state[i_mon]
            tmp_sim_PR_A = sim_PR_A[i_mon]
            tmp_sim_Tmax = sim_Tmax[i_mon]
            tmp_sim_Tmin = sim_Tmin[i_mon]
            tmp_sim_RSDS = sim_RSDS[i_mon]
            tmp_sim_EV = sim_EV[i_mon]

            simulation_length = month_lengths[Storyline_months[i_mon] - 1]
            # need to simulate entire month
            for e_day in range(0, simulation_length):
                # First determine if it is raining
                base_pr_state = tmp_sim_state[e_day]
                base_pr_A = tmp_sim_PR_A[e_day]
                base_tmax = tmp_sim_Tmax[e_day]
                base_tmin = tmp_sim_Tmin[e_day]
                base_rsds = tmp_sim_RSDS[e_day]
                base_EV = tmp_sim_EV[e_day]

                R_val = random.random()
                if base_pr_state == 'W':
                    static_probability = E_P_vec[0]
                    if R_val <= static_probability:
                        e_PR_state = 'W'
                    else:
                        e_PR_state = 'D'
                else:
                    static_probability = E_P_vec[3]
                    if R_val <= static_probability:
                        e_PR_state = 'D'
                    else:
                        e_PR_state = 'W'

                # Calculate precip if required
                if e_PR_state == 'W':
                    # Check PR flag
                    if E_P_F:
                        e_PR_A = np.abs(base_pr_A + percent2cdf(random.random(), E_P_CDF, E_P_bin))
                    else:
                        e_PR_A = percent2cdf(random.random(), E_P_CDF, E_P_bin)
                else:
                    e_PR_A = 0

                # Calculate other variables
                e_tmax = np.abs(base_tmax + percent2cdf(random.random(), E_CDF[0], E_bins[0]))
                e_tmin = np.abs(base_tmin + percent2cdf(random.random(), E_CDF[1], E_bins[1]))
                e_rsds = np.abs(base_rsds + percent2cdf(random.random(), E_CDF[2], E_bins[2]))
                e_EV = np.abs(base_EV + percent2cdf(random.random(), E_CDF[3], E_bins[3]))

                # Need to export data
                Full_e_PR_state.append(e_PR_state)
                Full_e_PR_A.append(e_PR_A)
                if e_tmax>e_tmin:
                    Full_e_tmax.append(e_tmax)
                    Full_e_tmin.append(e_tmin)
                else:
                    Full_e_tmax.append(e_tmin)
                    Full_e_tmin.append(e_tmax)
                Full_e_rsds.append(e_rsds)
                Full_e_ev.append(e_EV)

            sim_E_state.append(Full_e_PR_state)
            sim_E_PR_A.append(Full_e_PR_A)
            sim_E_Tmax.append(Full_e_tmax)
            sim_E_Tmin.append(Full_e_tmin)
            sim_E_RSDS.append(Full_e_rsds)
            sim_E_EV.append(Full_e_ev)
            sim_E_mon_vals.append([Storyline_months[i_mon]] * simulation_length)

        # Save the Extra Station data
        Tmp_extra_savename = Extra_sim_savename + '_P' + str(i_ex) + '_S' + str(sim_ind)
        print('Simulation ' + str(sim_ind + 1) + ' out of ' + str(
            number_of_simulations) + ' complete for extra station ' + str(i_ex + 1))

        if netcdf_flag:
            netcdf_saver(sim_E_PR_A, sim_E_Tmax, sim_E_Tmin, sim_E_RSDS, sim_E_EV, sim_E_mon_vals, Tmp_extra_savename,
                         Extra_site_lat[i_ex], Extra_site_lon[i_ex])
        else:
            np.savez(Tmp_extra_savename, PR_State=sim_E_state, PR_Amount=sim_E_PR_A,
                     Tmax=sim_E_Tmax, Tmin=sim_E_Tmin, RSDS=sim_E_RSDS, EV=sim_E_EV)
