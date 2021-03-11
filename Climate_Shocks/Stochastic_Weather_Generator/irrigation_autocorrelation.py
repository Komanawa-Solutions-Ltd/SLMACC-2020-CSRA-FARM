"""
 Author: Matt Hanson
 Created: 28/01/2021 11:15 AM
 """
from Climate_Shocks.get_past_record import get_restriction_record
import numpy as np
from scipy.stats import pearsonr, truncnorm
from Climate_Shocks.note_worthy_events.inverse_percentile import inverse_percentile


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

    data = get_restriction_record('detrended').reset_index()
    data = data.groupby(['year', 'month']).mean(False)

    for (y, m), frest in data.loc[:, ['f_rest', ]].itertuples(True, None):
        per, err = inverse_percentile(data.loc[(data.index.levels[0], m), 'f_rest'], frest)
        data.loc[(y, m), 'rest_per'] = per
        data.loc[(y, m), 'rest_per_err'] = err

    data.set_index('date').loc[:, ['rest_per', 'rest_per_err']].plot()

    fig, ax = plt.subplots()
    ax.set_title('rest_per')
    lags = 12
    auto = get_autocorrelation(data.rest_per, lags)
    ax.plot(range(lags), auto)
    fig, ax = plt.subplots()
    ax.set_title('restriction')
    auto = get_autocorrelation(data.f_rest, lags)
    ax.plot(range(lags), auto)
    plt.show()

    pass
