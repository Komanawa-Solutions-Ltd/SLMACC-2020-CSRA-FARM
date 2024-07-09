"""
 Author: Matt Hanson
 Created: 25/11/2020 11:42 AM
 """

import pandas as pd
import numpy as np
import os
import project_base

from komanawa.basgra_nz_py.supporting_functions.plotting import plot_multiple_results, plot_multiple_monthly_results
from Pasture_Growth_Modelling.initialisation_support.comparison_support import make_mean_comparison, \
    get_horarata_data_old, get_indicative_irrigated
from Pasture_Growth_Modelling.historical_average_baseline import run_past_basgra_dryland, run_past_basgra_irrigated




if __name__ == '__main__':
    outdir = project_base.slmmac_dir.joinpath(r"pasture_growth_modelling/historical_runs_v2")
    save = False
    data = {
        'irrigated_eyrewell': run_past_basgra_irrigated(),
        #'irrigated_oxford': run_past_basgra_irrigated(site='oxford'),
        'dryland_oxford': run_past_basgra_dryland(site='oxford'),
    }
    for i, k in enumerate(data.keys()):
        data[k].loc[:, 'RESEEDED'] += i  # any more fo these to raise up to see with multiple runs

    data2 = {e: make_mean_comparison(v, 'mean') for e, v in data.items()}
    data2['Horoata'] = get_horarata_data_old()
    data2['indicative_irr'] = get_indicative_irrigated()

    out_vars = ['DM', 'DMH', 'YIELD', 'DMH_RYE', 'DM_RYE_RM', 'DMH_WEED', 'DM_WEED_RM', 'IRRIG', 'RAIN', 'EVAP', 'TRAN',
                'per_PAW', 'pg', 'RESEEDED',
                'pga_norm', 'BASAL']

    data3 = {e: v.groupby('month').mean() for e, v in data.items()}
    plt_outdir_sim = None
    plt_outdir_aver_yr = None
    if save:
        plt_outdir_sim = os.path.join(outdir, 'plots', 'full_hist')
        plt_outdir_aver_yr = os.path.join(outdir, 'plots', 'aver_yr')
        for d in [plt_outdir_aver_yr, plt_outdir_sim]:
            if not os.path.exists(d):
                os.makedirs(d)
        _org_describe_names = ['count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max']
        outkeys = ['pga', 'pga_norm']
        outdata = pd.DataFrame(index=pd.Series(range(1, 13), name='month'),
                               columns=pd.MultiIndex.from_product([data.keys(),
                                                                   ['pga', 'pga_norm'],
                                                                   _org_describe_names]), dtype=float)
        for k, v in data.items():
            temp = v.groupby('month').describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            outdata.loc[:, (k, outkeys, _org_describe_names)] = temp.loc[:, (outkeys, _org_describe_names)].values
            v.to_csv(os.path.join(outdir, '{}_raw.csv'.format(k)))
            v.resample('10D').mean().to_csv(os.path.join(outdir, '{}_10daily.csv'.format(k)))
            v.resample('M').mean().to_csv(os.path.join(outdir, '{}_monthly.csv'.format(k)))
            v.groupby('month').mean().to_csv(os.path.join(outdir, '{}_average_year.csv'.format(k)))
        outdata = outdata.loc[:, (data.keys(),outkeys, _org_describe_names)]
        outdata.round(3).to_csv(os.path.join(outdir, 'all_sim_pga_pganorm_desc.csv'))

    # make plots
    plot_multiple_results(data=data, out_vars=out_vars, rolling=90, label_rolling=True, label_main=False,
                          main_kwargs={'alpha': 0.2},
                          show=False, outdir=plt_outdir_sim, title_str='historical_')
    plot_multiple_monthly_results(data=data3, out_vars=out_vars, show=False, outdir=plt_outdir_aver_yr,
                                  title_str='average_year_', main_kwargs={'marker': 'o'})
    plot_multiple_monthly_results(data=data2, out_vars=['pg', 'pgr'], show=(not save), outdir=plt_outdir_aver_yr,
                                  title_str='cumulative_average_year_', main_kwargs={'marker': 'o'})
