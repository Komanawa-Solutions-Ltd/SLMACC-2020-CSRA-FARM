"""
modified from:
http://earthpy.org/flow.html
created matt_dumont 
on: 16/06/22
"""
import numpy as np
import scipy.stats as sp
import scipy.optimize as op


def fdc(data):
    '''
    Generate flow duration curve for hydrologic time series data

    df = pandas dataframe containing data
    site = column within dataframe that contains the flow values
    begyear = start year for analysis
    endyear = end year for analysis
    '''

    data = np.sort(data)
    ranks = sp.rankdata(data, method='average')
    ranks = ranks[::-1]
    prob = [100 * (ranks[i] / (len(data) + 1)) for i in range(len(data))]
    return prob, data


def fdcmatch(data, normalizer=1, fun=1):
    '''
    * This function creates a flow duration curve (or its inverse) and then matches a natural logrithmic function (or its inverse - exp)
    to the flow duration curve
    * The flow duration curve will be framed for averaged daily data for the duration of one year (366 days)

    PARAMETERS:
        data = flow data
        normalizer = value to use to normalize discharge; defaults to 1 (no normalization)
        fun = 1 for probability as a function of discharge; 0 for discharge as a function of probability; default=1
            * 1 will choose:
                prob = a*ln(discharge*b+c)+d
            * 0 will choose:
                discharge = a*exp(prob*b+c)+d
    RETURNS:
        para, parb, parc, pard, r_squared_value, stderr

        par = modifying variables for functions = a,b,c,d
        r_squared_value = r squared value for model
        stderr = standard error of the estimate

    REQUIREMENTS:
        pandas, scipy, numpy
    '''
    data = np.sort(data)
    data = [(data[i]) / normalizer for i in range(len(data))]

    # ranks data from smallest to largest
    ranks = sp.rankdata(data, method='average')

    # reverses rank order
    ranks = ranks[::-1]

    # calculate probability of each rank
    prob = [(ranks[i] / (len(data) + 1)) for i in range(len(data))]

    # choose which function to use
    if fun == 1:
        # function to determine probability as a function of discharge
        def func(x, a, b, c, d):
            return a * np.log(x * b + c) + d

        # matches func to data
        par, cov = op.curve_fit(func, data, prob)

        # checks fit of curve match
        slope, interecept, r_value, p_value, stderr = \
            sp.linregress(prob, [par[0] * np.log(data[i] * par[1] + par[2]) + par[3] for i in range(len(data))])
    else:
        # function to determine discharge as a function of probability
        def func(x, a, b, c, d):
            return a * np.exp(x * b + c) + d

        # matches func to data
        par, cov = op.curve_fit(func, prob, data)

        # checks fit of curve match
        slope, interecept, r_value, p_value, stderr = \
            sp.linregress(data, [par[0] * np.exp(prob[i] * par[1] + par[2]) + par[3] for i in range(len(prob))])

    # return function, parameters (a,b,c,d), r-squared of model fit, and standard error of model fit
    return func, par, round(r_value ** 2, 2), round(stderr, 5)
