"""
created matt_dumont 
on: 7/16/24
"""
import numpy as np
import matplotlib.pyplot as plt
from komanawa.slmacc_csra import get_eyrewell_historical_detrended
from komanawa.simple_farm_model.simple_dairy_model import DairyModelWithSCScarcity


def run_historical_farm_model(s, a, b, c):
    """
    :param s: scale - the maximum value of the curve (if s=1, the maximum value is 1)
    :param a: steepness - smoothing parameter as a increases the curve becomes steeper and the inflection point moves to the right
    :param b: steepness about the inflection point
    :param c: inflection point (if a=1, c is the x value at which y=0.5)
    """
    history = get_eyrewell_historical_detrended()
    history['water_year'] = history['year'] + (history['month'] >= 7).astype(int)
    history = history.loc[(history['water_year'] < 2020) & (history['water_year'] > 1973)]
    history = history.groupby(['water_year', 'month']).mean()
    mj_per_kg_dm = 11  # MJ ME /kg DM
    sup_cost = 406 / 1000 / mj_per_kg_dm
    product_price = 8.09
    all_months = np.array([(7 - 1 + i) % 12 + 1 for i in range(12)])
    inpg = np.array([history.loc[wy].loc[all_months, 'pg'].values for wy in history.index.levels[0]]).transpose()
    nsims = inpg.shape[1]

    model = DairyModelWithSCScarcity(all_months,
                                     istate=[0] * nsims, pg=inpg, ifeed=[0] * nsims, imoney=[0] * nsims,
                                     sup_feed_cost=sup_cost,
                                     product_price=product_price,
                                     monthly_input=True,
                                     s=s, a=a, b=b, c=c)

    model.run_model(printi=False)
    needed_feed = model.get_annual_feed()
    imported_feed = model.cum_feed_import[-1]
    percent_imported = imported_feed / needed_feed * 100
    model.plot_scurve(True)

    # todo could also optimise peak cow, and annual profit...
    return percent_imported, model.model_money[-1]


if __name__ == '__main__':
    outputs = run_historical_farm_model(s=10, a=1, b=0.9, c=16)
    fig, axs = plt.subplots(2)
    axs[0].hist(outputs[0], bins=20)
    axs[0].set_title('Percent of feed imported')
    axs[1].hist(outputs[1], bins=20)
    axs[1].set_title('Annual profit')
    plt.show()
