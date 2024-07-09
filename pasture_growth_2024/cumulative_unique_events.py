"""
created matt_dumont 
on: 7/9/24
"""
import matplotlib.pyplot as plt
import numpy as np
from komanawa.slmacc_csra import get_normalised_unique_event_data
from komanawa.slmacc_csra.base_info import default_mode_sites
from Storylines.storyline_params import month_len
import project_base

savedir = project_base.slmmac_dir.joinpath('0_Y2_and_Final_Reporting', 'final_plots', 'unique_events')
savedir.mkdir(exist_ok=True)


def make_cumulative_event_tables(site, mode, inc_irr_rest):
    data = get_normalised_unique_event_data(site, mode)
    datamonth = data.month.values[:, np.newaxis] + np.arange(24)[np.newaxis, :]
    datamonth[datamonth > 12] -= 12
    datamonth[datamonth > 12] -= 12
    ndays = np.array([month_len[m] for m in datamonth.flatten()]).reshape(datamonth.shape)
    data['annual_impact'] = (data[np.arange(12).astype(str)] * ndays[:, :12]).sum(axis=1)
    if not inc_irr_rest and mode != 'dryland':
        data = data.loc[data.rest == 50]
    data = data[['month', 'precip', 'temp', 'rest', 'annual_impact']]
    data = data.sort_values('annual_impact')
    return data


def plot_event_impacts():
    fig, (ax, ax_no_irr) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, sharey=True)
    colors = {
        ('eyrewell', 'irrigated'): 'red',
        ('oxford', 'irrigated'): 'orange',
        ('oxford', 'dryland'): 'blue',
    }
    for inc_irr, ax in zip([True, False], [ax, ax_no_irr]):
        for (site, mode), color in colors.items():
            data = make_cumulative_event_tables(site, mode, inc_irr)
            data = data.reset_index()
            ax.plot(data.index / data.index.max(), data.annual_impact / 1000, label=f'{site}-{mode}', color=color)
        ax.set_title(f'Unique Events {"Including" if inc_irr else "Excluding"} Irrigation Restrictions')
        ax.axhline(0, color='black', linestyle='--')
        ax.legend()

    fig.supxlabel('Event in order of impact (normalised)')
    fig.supylabel('Impact (Tons DM/ha/year)')
    fig.tight_layout()
    fig.savefig(savedir.joinpath('all_unique_event_impacts.png'))

def export_tables():
    for mode, site in default_mode_sites:
        for inc_irr in [True, False]:
            data = make_cumulative_event_tables(site, mode, inc_irr)
            data.to_csv(savedir.joinpath(f'{site}-{mode}-{"inc" if inc_irr else "exc"}_irr.csv'))


if __name__ == '__main__':
    plot_event_impacts()
    export_tables()

