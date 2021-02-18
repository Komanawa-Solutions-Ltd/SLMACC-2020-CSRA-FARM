"""
 Author: Matt Hanson
 Created: 18/02/2021 1:20 PM
 """

from Storylines.check_storyline import get_acceptable_events, get_months_with_events, ensure_no_impossible_events
from Climate_Shocks.climate_shocks_env import storyline_dir
import pandas as pd
import os
from BS_work.SWG.SWG_wrapper import *

filenm = 'make-npzs.csv'
def write_all_story():
    acc = get_acceptable_events()
    years, months, temps, precips = [], [], [], []
    year = 2000
    for k, v in acc.items():
        for m in [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]:
            if m == 1:
                year += 1
            years.append(year)
            months.append(m)
            if m in v:
                temps.append(k.split('-')[0])
                precips.append(k.split('-')[1])
            elif m == 7:
                temps.append('C')
                precips.append('A')
            else:
                temps.append('A')
                precips.append('A')
    outdata = pd.DataFrame(index=pd.date_range('2000-07-01', periods=len(years), freq='MS'))
    outdata.loc[:, 'year'] = years
    outdata.loc[:, 'month'] = months
    outdata.loc[:, 'precip_class'] = precips
    outdata.loc[:, 'temp_class'] = temps
    outdata.loc[:, 'rest'] = 0

    ensure_no_impossible_events(outdata)
    outdata.to_csv(os.path.join(storyline_dir, filenm))


def run_mk_npz(base_dir, vcf): #todo run before running a big suite
    outdir = os.path.join(ksl_env.slmmac_dir_unbacked, 'SWG_runs', 'make-all-npz')
    yml = os.path.join(outdir, 'test.yml')
    create_yaml(outpath_yml=yml, outsim_dir=outdir,
                nsims=100,
                storyline_path=os.path.join(storyline_dir, filenm),
                base_dir=base_dir,
                sim_name=None,
                xlat=oxford_lat, xlon=oxford_lon,
                vcf=vcf)
    temp = run_SWG(yml, outdir, rm_npz=True, clean=False)


if __name__ == '__main__':
    write_all_story()
