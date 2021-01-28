import numpy as np
from Climate_Shocks.get_past_record import get_restriction_record
import matplotlib.pyplot as plt
import scipy.stats

# generally, options:
# 1 resample previous time series (flow or restriction), start here
# 2 make sythetic flow data and translate to restriction
# 3 make sytheic restriction data by creating autocorrelated data...

# either make the data directly or make a large set that covers all restriction options and then sample from that.


# todo this is incredible https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014WR016827
#todo this is an option: https://github.com/julianneq/Kirsch-Nowak_Streamflow_Generator
#todo this coudl be simple https://eranraviv.com/bootstrapping-time-series-r-code/
#todo https://waterprogramming.wordpress.com/2017/08/29/open-source-streamflow-generator-part-i-synthetic-generation/
#todo https://waterprogramming.wordpress.com/2017/02/07/synthetic-streamflow-generation/
#todo https://www.researchgate.net/publication/222797906_A_nonparametric_stochastic_approach_for_multisite_disaggregation_of_annual_to_daily_streamflow
#todo https://www.mdpi.com/2306-5338/5/4/66/htm
#todo https://www.sciencedirect.com/science/article/pii/S1364815219307133


# below from
#https://stackoverflow.com/questions/33898665/python-generate-array-of-specific-autocorrelation

def sample_signal(n_samples, corr, mu=0, sigma=1):
    """

    :param n_samples: number of samples
    :param corr: desired auto-correlation at lag one
    :param mu: mean
    :param sigma: standard deviation
    :return:
    """
    assert 0 < corr < 1, "Auto-correlation must be between 0 and 1"

    # Find out the offset `c` and the std of the white noise `sigma_e`
    # that produce a signal with the desired mean and variance.
    # See https://en.wikipedia.org/wiki/Autoregressive_model
    # under section "Example: An AR(1) process".
    c = mu * (1 - corr)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c + np.random.normal(0, sigma_e)]
    for _ in range(1, n_samples):
        signal.append(c + corr * signal[-1] + np.random.normal(0, sigma_e))

    return np.array(signal)

def compute_corr_lag_1(signal):
    return np.corrcoef(signal[:-1], signal[1:])[0][1]

def examples():
# Examples.
    print(compute_corr_lag_1(sample_signal(5000, 0.5)))
    print(np.mean(sample_signal(5000, 0.5, mu=2)))
    print(np.std(sample_signal(5000, 0.5, sigma=3)))

# explore data (matt)
def explore_data():
    data = get_restriction_record()
    for k in ['f_rest', 'flow']:
        for month in [9,10,11,12,1,2,3,4,5]:
            fig, ax = plt.subplots()
            plt_data = data.loc[(data.month==month),k]
            if k =='flow':
                plt_data = np.log10(plt_data)
            ax.hist(plt_data, bins=50)
            fig, ax_ind = plt.subplots()
            ax.set_title('{} all, month {}'.format(k,month))
            ax_ind.set_title('{} unique, month {}'.format(k,month))
            for y in data.year.unique():
                plt_data = data.loc[(data.month==month)&(data.year==y),k]
                if k =='flow':
                    plt_data = np.log10(plt_data)
                ax_ind.hist(plt_data, alpha=0.5)

    plt.show()

    pass

def plot_pdf(values):
    raise NotImplementedError


if __name__ == '__main__':
    explore_data()