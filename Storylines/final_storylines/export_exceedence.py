"""
created matt_dumont 
on: 12/07/22
"""
from pathlib import Path
import project_base
import shutil

if __name__ == '__main__':
    outdir = Path().home().joinpath('Downloads/exceed')
    outdir.mkdir(exist_ok=True)
    base = Path(project_base.slmmac_dir).joinpath("outputs_for_ws", "norm",
                                             "random_scen_plots", "1yr_correct")
    paths = base.glob("*-*_1yr_cumulative_exceed_prob.csv")
    for p in paths:
        shutil.copyfile(p, outdir.joinpath(p.name.replace('_1yr', '-mod_1yr')))

    base = Path(project_base.slmmac_dir).joinpath("outputs_for_ws", "norm",
                                             "random_scen_plots", "1yr")
    paths = base.glob("*-*_1yr_cumulative_exceed_prob.csv")
    for p in paths:
        shutil.copyfile(p, outdir.joinpath(p.name.replace('_1yr', '-raw_1yr')))
