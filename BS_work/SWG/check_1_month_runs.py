"""
one off testing script not a part of the processs!
 Author: Matt Hanson
 Created: 22/02/2021 9:42 AM
 """
import os
import ksl_env
import yaml
import pandas as pd
from Climate_Shocks import climate_shocks_env
from BS_work.SWG.SWG_wrapper import _check_data_v1, _get_possible_months
oxford_lat, oxford_lon = -43.296, 172.192
swg = os.path.join(os.path.dirname(__file__), 'SWG_Final.py')
default_vcf = os.path.join(os.path.dirname(__file__), 'event_definition_data_fixed.csv')

def make_event_prob(base_dir, outpath=os.path.join(climate_shocks_env.supporting_data_dir, 'prob_gen_event_swg.csv')):
    """
    make probabilty that swg will not create a correct realisation
    :param base_dir:
    :param outpath:
    :return:
    """
    out_dict = {}
    cold_months, wet_months, hot_months, dry_months = _get_possible_months()
    for d in os.listdir(base_dir):
        if '.' in d:
            continue
        print(d)
        swg_dir = os.path.join(base_dir, d)
        yml_path = os.path.join(swg_dir, 'ind.yml')
        with open(yml_path, 'r') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            param_dict = yaml.load(file, Loader=yaml.FullLoader)
        storyline_path = param_dict['story_line_filename']
        storyline = pd.read_csv(storyline_path)
        paths = pd.Series(os.listdir(swg_dir))
        paths = paths.loc[(~paths.str.contains('exsites')) & (paths.str.contains('.nc'))]
        outdata = pd.DataFrame(index=paths, columns=['count'])
        for p in paths:
            m = int(p.split('-')[0].replace('m', ''))
            fp = os.path.join(swg_dir, p)
            num_dif, out_keys, hot, cold, wet, dry = _check_data_v1(fp, storyline, m, cold_months, wet_months,
                                                                    hot_months, dry_months, return_full_results=True)
            outdata.loc[p, 'count'] = num_dif
            for k in ['hot', 'cold', 'wet', 'dry']:
                outdata.loc[p, k] = eval(k)
        outdata.to_csv(os.path.join(swg_dir, 'num_diff.csv'))
        out_dict[d] = outdata.loc[:, 'count'].mean()
    pd.Series(out_dict).to_csv(outpath)



if __name__ == '__main__':
    base_dir = os.path.join(ksl_env.unbacked_dir, 'SWG_runs', 'try_individual_nPrecip')
    out_dict = {}
    for d in os.listdir(base_dir):
        if '.csv' in d:
            continue
        print(d)
        outdir = os.path.join(base_dir, d)
        yml_path = os.path.join(outdir, 'ind.yml')
        paths = pd.Series(os.listdir(outdir))
        paths = paths.loc[(~paths.str.contains('exsites')) & (paths.str.contains('.nc'))]
        outdata = pd.DataFrame(index=paths, columns=['count'])
        for p in paths:
            num_dif, out_keys, hot, cold, wet, dry = check_single(os.path.join(outdir, p), yml_path,
                                                                  m=int(p.split('-')[0].replace('m', '')))
            outdata.loc[p, 'count'] = num_dif
            for k in ['hot', 'cold', 'wet', 'dry']:
                outdata.loc[p, k] = eval(k)
        outdata.to_csv(os.path.join(outdir, 'num_diff.csv'))
        out_dict[d] = outdata.loc[:, 'count'].mean()
    pd.Series(out_dict).to_csv(os.path.join(base_dir, 'event_overview.csv'))
