"""
 Author: Matt Hanson
 Created: 6/07/2021 8:57 AM
 """
import os.path

import ksl_env
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite, default_mode_sites
from Storylines.storyline_evaluation.plot_nyr_suite import plot_impact_for_sites
import matplotlib.pyplot as plt


def plot_all_site_v_site(nyrs):
    data = {}
    for y in nyrs:
        outdir = os.path.join(ksl_env.slmmac_dir, f"outputs_for_ws/norm/random_scen_plots/{y}yr")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if y == 1:
            temp_data = get_1yr_data()
            for mode, site in default_mode_sites:
                data[f'{site}-{mode}'] = temp_data.loc[:, f'{site}-{mode}_pg_yr1'] / 1000
        else:
            for mode, site in default_mode_sites:
                temp_data = get_nyr_suite(y, site, mode)
                data[f'{site}-{mode}'] = temp_data.loc[:, f'{site}-{mode}_pg_yr{y}'] / 1000
                pass
        figs, figids = plot_impact_for_sites(data, 300, (14, 7))
        for fig, fid in zip(figs, figids):
            fig.suptitle(f'Full Suite {y} year sim')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'site_comparison_all_{fid}.png'))


if __name__ == '__main__':

    plot_all_site_v_site([1, 2, 3, 5, 10])
