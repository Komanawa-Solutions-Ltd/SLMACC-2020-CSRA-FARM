"""
 Author: Matt Hanson
 Created: 23/12/2020 10:52 AM
 """
import pandas as pd
import ksl_env
import os
import matplotlib.pyplot as plt
from Climate_Shocks.note_worthy_events.inital_event_recurance import joint_hot_dry


def plot_data():
    full_event_names, joint_data = joint_hot_dry()
    org_prob = {
        4: 35.40,
        5: 27.10,
        8: 2.10,
        9: 12.50,
    }
    org_dry = {
        4: -34.30,
        5: -55.40,
        8: -30.80,
        9: -4.00,
    }
    org_irr = {
        4: -17.70,
        5: -19.30,
        8: -11.50,
        9: 13.40,
    }

    for m in [4, 5, 8, 9]:
        fig, ax = plt.subplots()
        ax.set_title('month: {}'.format(m))
        ax.set_xlabel('prob %')
        ax.set_ylabel('impact %')
        for k in full_event_names:
            x, y = joint_data.loc[m, (k, 'prob')], joint_data.loc[m, (k, 'mean_dry')]
            ax.scatter(x, y, c='darkred')
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            y = joint_data.loc[m, (k, 'mean_irr')]
            ax.scatter(x, y, c='darkblue')
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        ax.scatter(org_prob[m], org_dry[m], c='r', s=80, label='dryland')
        ax.scatter(org_prob[m], org_irr[m], c='b', s=80, label='irrigated')
        ax.legend()
    plt.show()  # todo spot check then iterate!


if __name__ == '__main__':
    plot_data()
