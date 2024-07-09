import project_base
import os
import pandas as pd
from Climate_Shocks.get_past_record import get_vcsn_record, event_def_path
import matplotlib.pyplot as plt

from komanawa.basgra_nz_py.basgra_python import run_basgra_nz
from komanawa.basgra_nz_py.supporting_functions.plotting import plot_multiple_results
from komanawa.basgra_nz_py.supporting_functions.woodward_2020_params import get_woodward_mean_full_params
from komanawa.basgra_nz_py.input_output_keys import matrix_weather_keys_pet

if __name__ == '__main__':
    detrend = pd.read_csv(os.path.join(os.path.dirname(event_def_path), 'daily_percentiles_detrended_v2.csv'))
    trend = pd.read_csv(os.path.join(os.path.dirname(event_def_path), 'daily_percentiles.csv'))
    for d in [detrend, trend]:
        d.loc[:, 'date'] = pd.to_datetime(d.loc[:, 'date'])
        d.loc[:, 'month'] = d.loc[:, 'date'].dt.month
        d.set_index('date', inplace=True)
    data = {
        'detrend': detrend,
        'trend': trend,
    }
    out_vars = ['hot_per', 'cold_per', 'dry_per', 'wet_per', ]
    plot_multiple_results(data, out_vars=out_vars)
    vcsn2 = get_vcsn_record('detrended2')
    vcsn2.loc[vcsn2.tmax >= 25, 'hotday'] = 1
    vcsn2 = vcsn2.groupby('year').sum()
    vcsn = get_vcsn_record()
    vcsn.loc[vcsn.tmax >= 25, 'hotday'] = 1
    vcsn = vcsn.groupby('year').sum()
    data = {'trend': vcsn,
            'detrend': vcsn2}
    plot_multiple_results(data, out_vars=['hotday'])

    data = {
        'trend': get_vcsn_record(),
        'detrend2': get_vcsn_record('detrended2')
    }
    diff = detrend - trend
    diff.loc[:, 'month'] = detrend.loc[:, 'month']
    dif2 = diff.groupby('month').mean()
    diff.to_csv(r"C:\Users\Matt Hanson\Downloads\detrend-trend_dif_raw.csv")
    dif2.to_csv(r"C:\Users\Matt Hanson\Downloads\detrend-trend_dif_monthly.csv")

    out_vars = ['doy', 'pet', 'radn', 'tmax', 'tmin', 'rain']
    plot_multiple_results(data, out_vars=out_vars, rolling=5,
                          main_kwargs={'alpha': 0.5})
