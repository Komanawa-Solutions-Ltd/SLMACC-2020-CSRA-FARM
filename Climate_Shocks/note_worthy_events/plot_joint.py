"""
 Author: Matt Hanson
 Created: 23/12/2020 10:52 AM
 """
import pandas as pd
import ksl_env
import os
import matplotlib.pyplot as plt
from Climate_Shocks.note_worthy_events.inital_event_recurance import joint_hot_dry, calc_hot_recurance_variable, \
    calc_dry_recurance_ndays, calc_dry_rolling


def plot_data_all():
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

def plot_data():
    full_event_names, joint_data = calc_dry_rolling()
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
        ax.set_title('month: {}mean'.format(m))
        ax.set_xlabel('prob %')
        ax.set_ylabel('impact %')
        for k in full_event_names:
            x, y = joint_data.loc[m, (k, 'prob')], joint_data.loc[m, (k, 'pga_dry_mean')]
            ax.scatter(x, y, c='darkred')
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            y = joint_data.loc[m, (k, 'pga_irr_mean')]
            ax.scatter(x, y, c='darkblue')
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        ax.scatter(org_prob[m], org_dry[m], c='r', s=80, label='dryland')
        ax.scatter(org_prob[m], org_irr[m], c='b', s=80, label='irrigated')
        ax.legend()

def plot_data_median():
    full_event_names, joint_data = calc_dry_rolling()
    org_prob = {
        4: 35.40,
        5: 27.10,
        8: 2.10,
        9: 12.50,
    }
    org_dry = {
        4: -33.80,
        5: -66.80,
        8: -30.80,
        9: 8.10,
    }
    org_irr = {
        4: -1.40,
        5: -18.00,
        8: -0.20,
        9: 16.80,
    }

    for m in [4, 5, 8, 9]:
        fig, ax = plt.subplots()
        ax.set_title('month: {} median'.format(m))
        ax.set_xlabel('prob %')
        ax.set_ylabel('impact %')
        for k in full_event_names:
            x, y = joint_data.loc[m, (k, 'prob')], joint_data.loc[m, (k, 'pga_dry_50%')]
            ax.scatter(x, y, c='darkred')
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            y = joint_data.loc[m, (k, 'pga_irr_50%')]
            ax.scatter(x, y, c='darkblue')
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        ax.scatter(org_prob[m], org_dry[m], c='r', s=80, label='dryland')
        ax.scatter(org_prob[m], org_irr[m], c='b', s=80, label='irrigated')
        ax.legend()


if __name__ == '__main__':
    calc_hot_recurance_variable()
    calc_dry_rolling()
    plot_data_median()
    plot_data()
    plt.show()
