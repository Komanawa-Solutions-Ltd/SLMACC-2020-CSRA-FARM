"""
 Author: Matt Hanson
 Created: 2/03/2021 2:25 PM
 """
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import glob
import ksl_env
from Storylines.storyline_building_support import default_storyline_time
from Pasture_Growth_Modelling.full_model_implementation import out_variables

default_outvars = ['m_'+e for e in out_variables] + ['m_PGRA','m_PGRA_cum']


def export_sim_to_csvs(nc_paths, outdir, outvars=default_outvars, inc_storylines=True, agg_fun=np.nanmean):

    if not os.path.exists(os.path.join(outdir, 'storylines')):
        os.makedirs(os.path.join(outdir, 'storylines'))

    nc_paths = np.atleast_1d(nc_paths)
    names = [os.path.basename(os.path.splitext(e)[0]) for e in nc_paths]
    outdata = {}
    variable_desc = {}
    for v in outvars:
        t = pd.DataFrame(index=range(len(default_storyline_time)), columns=names)
        t.loc[:, 'month'] = default_storyline_time.month
        t.loc[:, 'year'] = default_storyline_time.year
        t = t.set_index(['year', 'month'])
        outdata[v] = t
    for p, nm in zip(nc_paths, names):
        data = nc.Dataset(p)
        year = np.array(data.variables['m_year'])
        month = np.array(data.variables['m_month'])
        for k in outvars:
            outdata[k].loc[zip(year, month), nm] = agg_fun(data.variables[k], axis=1)

            # get nc attributes
            variable_desc[k] = {e: data.variables[k].getncattr(e) for e in data.variables[k].ncattrs()}
        if inc_storylines:
            with open(os.path.join(outdir, 'storylines', f'{nm}_story.csv'),'w') as f:
                f.writelines(data.storyline)
        data.close()
    var_str = ''
    for k in outvars:
        outdata[k].transpose().to_csv(os.path.join(outdir, f'{k}.csv'))
        var_str = var_str + '{}:\n  * {}\n'.format(
            k, " \n  * ".join([f"{e}: {v}" for e,v in variable_desc[k].items()])
        )
    with open(os.path.join(outdir,'README.txt'),'w') as f:
        f.write(f"""
            
This is a simple readme to help users understand the data here.  These data are provided for ease of access
and to support Water Strategies Farmmax modelling.  If others are using these data then it is better to 
access the data via the NetCDF files. For more information please contact: 
    * Matt Hanson: Matt@komanawa.com
    * Zeb Etheridge: Zeb@komanawa.com

The data here contains the story lines (in the storyline folder) and the {agg_fun.__name__} of each of the variables
The storylines are defined as a precipitation state (precip_class) (Dry, Average, Wet) and a 
temperature state (temp_class) (Cold, Average, Hot) as well as a restriction class (rest). the restriction 
class is defined based on a fraction of the month spent under full restrictions (here 2 days of half 
restrictions are identical to 1 day of full restrictions).  The individual time series of restrictions are
derived from a moving block boot strapping of de-trended historical restriction data. Note restrictions are
not applied to dryland simulations.

Pasture growth modelling conducted with BASGRA_NZ_PY {ksl_env.basgra_version}, (https://github.com/Komanawa-Solutions-Ltd/BASGRA_NZ_PY)

# variable descriptions
{var_str}
            """
        )



def export_all_in_pattern(base_outdir, patterns, outvars=default_outvars, inc_storylines=True, agg_fun=np.nanmean):
    paths = []
    patterns = np.atleast_1d(patterns)
    for pattern in patterns:
        paths.extend(np.array(glob.glob(pattern)))
    paths = np.array(paths)
    base_paths = pd.Series([os.path.basename(p) for p in paths])
    for sm in ['eyrewell-irrigated', 'oxford-dryland', 'oxford-irrigated']:
        temp_paths = paths[base_paths.str.contains(sm)]
        export_sim_to_csvs(nc_paths=temp_paths, outdir=os.path.join(base_outdir, sm),
                           outvars=outvars, inc_storylines=inc_storylines, agg_fun=agg_fun)

