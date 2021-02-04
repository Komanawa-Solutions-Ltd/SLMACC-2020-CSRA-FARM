"""
 Author: Matt Hanson
 Created: 28/01/2021 11:15 AM
 """
from Climate_Shocks.get_past_record import get_restriction_record
import numpy as np
from scipy.stats import pearsonr, truncnorm


def get_autocorrelation(x, lags=30):
    data = []
    size = len(x)
    for l in range(lags):
        r, p = pearsonr(x[0:size - l], x[l:])
        data.append(r)
    return data


# I'm thinking of a variable block size...
# optimal block size seems somwhere between 5-11, so use a variable block size  mean 8, sd 1


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    myclip_a = 4
    myclip_b = 15
    my_mean = 8
    my_std = 1.5
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    x = truncnorm(a, b, loc=my_mean, scale=my_std).rvs(size=10000)  # .round().astype(int)
    fig, ax = plt.subplots()
    ax.hist(x, bins=2 * (15 - 4) + 1)

    lags = 30
    org_data = get_restriction_record('detrended')
    for m in range(1, 13):
        fig, ax = plt.subplots()
        ax.set_title('month: {}'.format(m))
        data = org_data.loc[org_data.month == m]
        data = data.f_rest.values
        auto = get_autocorrelation(data, lags)
        ax.plot(range(lags), auto)
    plt.show()
