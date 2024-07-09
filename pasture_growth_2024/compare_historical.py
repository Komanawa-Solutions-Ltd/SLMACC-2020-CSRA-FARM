"""
created matt_dumont 
on: 7/9/24
"""

# todo compare historical to 1 year data...

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from komanawa.slmacc_csra.base_info import default_mode_sites
from komanawa.slmacc_csra import get_historical_pg_data, get_historical_quantified_data, get_1yr_non_exceedence_prob


def plot_historical_vs_1yr():
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, figsize=(10, 8))
    leg_ax = axs[-1, -1]
    leg_ax.axis('off')
    fig.suptitle('Historical vs 1yr non-exceedance probability')
    fig.supxlabel('Pasture Growth (tons DM / ha / year)')
    fig.supylabel('Cumulative Probability (%)')
    colors = {  # todo choose better colors...
        'historical': 'blue',
        'historical quantized trended': 'red',
        'historical quantized de-trended': 'purple',
        'random suite': 'pink',
    }
    for j, site in enumerate(['oxford', 'eyrewell']):
        for i, mode in enumerate(['irrigated', 'store400', 'store600', 'store800', 'dryland']):
            ax = axs[i, j]
            if i == 0:
                ax.set_title(site.capitalize())
            if j == 0:
                ax.set_ylabel(mode.capitalize())
            if site == 'eyrewell' and mode == 'dryland':
                continue

    # todo legend...

    raise NotImplementedError


def plot_historical_vs_suite_nyear(nyr):  # todo after I see outputs of 1 year.
    raise NotImplementedError