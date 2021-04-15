"""
 Author: Matt Hanson
 Created: 15/04/2021 2:18 PM
 """
from Storylines.storyline_evaluation.plot_nyr_suite import *
from Storylines.storyline_runs.lauras_v2 import get_laura_v2_pg_prob
import ksl_env
from Storylines.storyline_building_support import default_mode_sites

def plot_1yr(save=False):
    for mode, site in default_mode_sites:
        outdir = None
        if save:
            outdir = os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'1yr')
        plot_all_nyr(site, mode, nyr=1, outdir=outdir, other_scen=None, other_scen_lbl='other storylines',
                     pt_labels=False)

    if not save:
        plt.show()
    else:
        plt.close('all')

def plot_3yr_no_additional(save=False):
    for mode, site in default_mode_sites:
        print(f'{site} - {mode}')
        outdir = None
        if save:
            outdir = os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'3yr')
        plot_all_nyr(site, mode, nscatter=int(7e4), nyr=3, outdir=outdir, other_scen=None,
                     other_scen_lbl='other storylines',
                     pt_labels=False)

        if not save:
            plt.show()
        else:
            plt.close('all')

def plot_3yr_additional(get_add_fun, other_scen_lbl, pt_labels=True, save=False):
    for mode, site in default_mode_sites:
        print(f'{site} - {mode}')
        additional = get_add_fun(site, mode)
        outdir = None
        if save:
            outdir = os.path.join(ksl_env.slmmac_dir, 'random_scen_plots', f'3yr_{other_scen_lbl}')
        plot_all_nyr(site, mode, nscatter=500, num=5, nyr=3, outdir=outdir, other_scen=additional,
                     other_scen_lbl=other_scen_lbl,
                     pt_labels=pt_labels, step_size=0.1, plt_data_density=True, close=True)

        if not save:
            plt.show()
        else:
            plt.close('all')

if __name__ == '__main__': # todo start here!
    plot_3yr_additional(get_laura_v2_pg_prob, 'lauras_v2', pt_labels=True, save=True)  # todo how long does this take
