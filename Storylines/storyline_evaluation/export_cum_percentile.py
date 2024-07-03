"""
 Author: Matt Hanson
 Created: 16/04/2021 11:26 AM
 """
import pandas as pd
import numpy as np
import os
import project_base
from Storylines.storyline_building_support import default_mode_sites
from Storylines.storyline_runs.run_random_suite import get_1yr_data, get_nyr_suite
from Storylines.storyline_evaluation.storyline_eval_support import calc_cumulative_impact_prob
from scipy.interpolate import interp1d


def export_cum_percentile(nyr, outdir, step_size=0.1):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for mode, site in default_mode_sites:
        if nyr == 1:
            data = get_1yr_data(bad_irr=True, good_irr=True)
        else:
            print('reading data')
            data = get_nyr_suite(nyr, site=site, mode=mode)
        data.dropna(inplace=True)
        if 'store' in mode:

            x = data.loc[:, f'log10_prob_irrigated']
        else:
            x = data.loc[:, f'log10_prob_{mode}']
        y = data[f'{site}-{mode}_pg_yr{nyr}'] / 1000

        prob = np.round(np.arange(0.01, 1, 0.01), 2)
        outdata = pd.DataFrame(index=prob)
        outdata.index.name = 'probability'
        cum_pgr, cum_prob = calc_cumulative_impact_prob(pgr=y,
                                                        prob=x, stepsize=step_size,
                                                        more_production_than=False)
        f = interp1d(cum_prob, cum_pgr)
        outdata.loc[:, 'non-exceedance_pg'] = f(prob)

        cum_pgr2, cum_prob2 = calc_cumulative_impact_prob(pgr=y,
                                                          prob=x, stepsize=step_size,
                                                          more_production_than=True)
        f = interp1d(cum_prob2, cum_pgr2)
        outdata.loc[:, 'exceedance_pg'] = f(prob)

        outdata.to_csv(os.path.join(outdir, f'{site}-{mode}_cumulative_prob.csv'))


if __name__ == '__main__':
    export_cum_percentile(1, outdir=os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'1yr'))
    export_cum_percentile(2, outdir=os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'2yr'))
    export_cum_percentile(3, outdir=os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'3yr'))
    export_cum_percentile(5, outdir=os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'5yr'))
    export_cum_percentile(10, outdir=os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'10yr'))
