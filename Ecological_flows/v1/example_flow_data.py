"""
 Author: Matt Hanson
 Created: 12/10/2021 11:15 AM
 """
import pandas as pd

import project_base
from Ecological_flows.v1.river_flow_pgr_model import get_rest_river_output_from_storyline_path, make_current_restrictions
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import ticker

outdir = os.path.join(ksl_env.slmmac_dir, 'ecological_flows', 'example_bad_river')
os.makedirs(outdir, exist_ok=True)


def make_example_data(num, seed=1154):
    storylines_paths = glob.glob(os.path.join(ksl_env.slmmac_dir,
                                              r"outputs_for_ws\norm\possible_final_stories\bad_stories_eyrewell",
                                              "storylines_cluster_*",
                                              "rsl*.csv"))
    np.random.seed(seed)
    use_storyline_paths = np.random.choice(storylines_paths, num, replace=False)
    np.random.seed(seed)
    use_seeds = np.random.randint(5, 564832, num)
    nruns = 2
    cmap = get_cmap('tab10')
    colors = [cmap(i / nruns) for i in range(nruns)]

    for sp, us in zip(use_storyline_paths, use_seeds):
        name = os.path.basename(sp).replace('.csv', '')
        pd.read_csv(sp).to_csv(os.path.join(outdir, f'{name}_story.csv'))
        data = get_rest_river_output_from_storyline_path(sp, nruns, make_current_restrictions, us)
        data.to_csv(os.path.join(outdir, f'{name}_flow.csv'))
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax1.set_ylabel('log 10 flow')
        ax2.set_ylabel('restriction')
        ax2.set_ylabel('date')
        for i in range(nruns):
            ax1.plot(data.index, data.loc[:, f'flow_{i:03d}'], c=colors[i])
            ax2.plot(data.index, data.loc[:, f'rest_{i:03d}'], c=colors[i])

        fig.suptitle(os.path.basename(sp))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'{name}_flow_log.png'))
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
        for i in range(nruns):
            ax1.plot(data.index, data.loc[:, f'flow_{i:03d}'], c=colors[i])
            ax2.plot(data.index, data.loc[:, f'rest_{i:03d}'], c=colors[i])
            ax1.set_ylabel('flow')
            ax2.set_ylabel('restriction')
            ax2.set_ylabel('date')
        fig.suptitle(os.path.basename(sp))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'{name}_flow_lin.png'))


if __name__ == '__main__':
    make_example_data(15)
