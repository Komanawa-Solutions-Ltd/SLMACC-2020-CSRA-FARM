"""
 Author: Matt Hanson
 Created: 5/07/2021 5:01 PM
 """
import pandas as pd

from Storylines.storyline_evaluation.storyline_characteristics_for_impact import storyline_subclusters, \
    get_month_limits_from_most_probable, default_mode_sites
import os
import ksl_env

base_dir = os.path.join(ksl_env.slmmac_dir, r"outputs_for_ws\norm\possible_final_stories")

# bounds are in fraction of new year.
eyrewell_irr_l = 12.050 * 1000
eyrewell_irr_u = 13.750 * 1000

oxford_irr_l = 9.67 * 1000
oxford_irr_u = 10.9 * 1000

oxford_dry_l = 2.675 * 1000
oxford_dry_u = 3.8 * 1000


def autumn_drought_1():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought1'),
        lower_bound={
            'oxford-dryland': oxford_dry_l,
            'eyrewell-irrigated': eyrewell_irr_l,
            'oxford-irrigated': oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': oxford_dry_u,
            'eyrewell-irrigated': eyrewell_irr_u,
            'oxford-irrigated': oxford_irr_u,

        },

        state_limits={
            2: (['D'], '*', (0.5, 1)),
            3: (['D'], '*', (0.5, 1)),
            4: (['D'], '*', (0.5, 1)),

        },
        n_clusters=5,
        n_pcs=15,
        save_stories=True, correct=True
    )


def autumn_drought_2():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought2'),
        lower_bound={
            'oxford-dryland': oxford_dry_l,
            'eyrewell-irrigated': eyrewell_irr_l,
            'oxford-irrigated': oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': oxford_dry_u,
            'eyrewell-irrigated': eyrewell_irr_u,
            'oxford-irrigated': oxford_irr_u,

        },

        state_limits={
            2: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


def autumn_drought_1_no_tresh():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought1_no_thresh'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,

        },

        state_limits={
            2: (['D'], '*', (0.5, 1)),
            3: (['D'], '*', (0.5, 1)),
            4: (['D'], '*', (0.5, 1)),

        },
        n_clusters=5,
        n_pcs=15,
        save_stories=True, correct=True
    )


def autumn_drought_2_no_thresh():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought2_no_thresh'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,

        },

        state_limits={
            2: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


def autumn_drought_3_no_thresh():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought3_no_thresh'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,

        },

        state_limits={
            2: (['D'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


thresh = 0.05
thresh_months = [6, 7, 8, 9, 10, 11, 12]


def autumn_drought_1_mon_thresh():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought1_mon_thresh'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,

        },

        state_limits={
            2: (['D'], '*', (0.5, 1)),
            3: (['D'], '*', (0.5, 1)),
            4: (['D'], '*', (0.5, 1)),

        },
        n_clusters=5,
        n_pcs=15,
        save_stories=True, correct=True,
        monthly_limits=get_month_limits_from_most_probable(eyrewell_irr={m: thresh for m in thresh_months},
                                                           oxford_irr={m: thresh for m in thresh_months},
                                                           oxford_dry=None,
                                                           correct=True)
    )


def autumn_drought_2_mon_thresh():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought2_mon_thresh'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,

        },

        state_limits={
            2: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True,
        monthly_limits=get_month_limits_from_most_probable(eyrewell_irr={m: thresh for m in thresh_months},
                                                           oxford_irr={m: thresh for m in thresh_months},
                                                           oxford_dry=None,
                                                           correct=True)

    )


def autumn_drought_3_mon_thresh():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought3_mon_thresh'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': None,
            'oxford-irrigated': None,

        },

        state_limits={
            2: (['D'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True,
        monthly_limits=get_month_limits_from_most_probable(eyrewell_irr={m: thresh for m in thresh_months},
                                                           oxford_irr={m: thresh for m in thresh_months},
                                                           oxford_dry=None,
                                                           correct=True)

    )


def dry_summer():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'dry_summer1'),
        lower_bound={
            'oxford-dryland': oxford_dry_l,
            'eyrewell-irrigated': eyrewell_irr_l,
            'oxford-irrigated': oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': oxford_dry_u,
            'eyrewell-irrigated': eyrewell_irr_u,
            'oxford-irrigated': oxford_irr_u,

        },

        state_limits={
            9: (['D', 'A'], '*', (0.5, 1)),
            10: (['D', 'A'], '*', (0.5, 1)),
            11: (['D', 'A'], '*', (0.5, 1)),
            12: (['D', 'A'], '*', (0.5, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


def dry_summer2():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'dry_summer2'),
        lower_bound={
            'oxford-dryland': oxford_dry_l,
            'eyrewell-irrigated': eyrewell_irr_l,
            'oxford-irrigated': oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': oxford_dry_u,
            'eyrewell-irrigated': eyrewell_irr_u,
            'oxford-irrigated': oxford_irr_u,

        },

        state_limits={
            9: (['D'], '*', (0.5, 1)),
            10: (['D'], '*', (0.5, 1)),
            11: (['D'], '*', (0.5, 1)),
            12: (['D'], '*', (0.5, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


def all_stories():
    # just to test bounds
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'all_stories_10-20'),
        lower_bound={
            'oxford-dryland': oxford_dry_l,
            'eyrewell-irrigated': eyrewell_irr_l,
            'oxford-irrigated': oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': oxford_dry_u,
            'eyrewell-irrigated': eyrewell_irr_u,
            'oxford-irrigated': oxford_irr_u,

        },

        state_limits=None,
        n_clusters=15,
        n_pcs=15,
        save_stories=True, correct=True
    )


def good_stories():
    # just to test bounds
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'good_stories'),
        lower_bound={
            'oxford-dryland': 5 * 1000,
            'eyrewell-irrigated': 15.5 * 1000,
            'oxford-irrigated': 11.5 * 1000,
        },
        upper_bound={
            'oxford-dryland': 7 * 1000,
            'eyrewell-irrigated': 16.2 * 1000,
            'oxford-irrigated': 12.5 * 1000,

        },

        state_limits=None,
        n_clusters=15,
        n_pcs=15,
        save_stories=True, correct=True
    )


dry_sp_au_eyrewell_irr_l = 13 * 1000
dry_sp_au_eyrewell_irr_u = 15 * 1000

dry_sp_au_oxford_irr_l = 9 * 1000
dry_sp_au_oxford_irr_u = 11 * 1000

dry_sp_au_oxford_dry_l = 4 * 1000
dry_sp_au_oxford_dry_u = 6 * 1000

# testing the limitations
dry_sp_au_eyrewell_irr_l = None
dry_sp_au_eyrewell_irr_u = None

dry_sp_au_oxford_irr_l = None
dry_sp_au_oxford_irr_u = None

dry_sp_au_oxford_dry_l = None
dry_sp_au_oxford_dry_u = None


def dry_spring_autumn1():
    # just to test bounds
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'dry_spring_autumn1'),
        lower_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_l,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_l,
            'oxford-irrigated': dry_sp_au_oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_u,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_u,
            'oxford-irrigated': dry_sp_au_oxford_irr_u,

        },

        state_limits={
            9: (['D'], '*', '*'),
            10: (['D'], '*', '*'),
            11: ('*', '*', (0.5, 1)),
            12: ('*', '*', (0.5, 1)),
            1: (['D', 'A'], ['H', 'A'], (0.5, 1)),
            2: (['D', 'A'], ['H', 'A'], (0.5, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.5, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.5, 1)),

        },

        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


def dry_spring_autumn2():
    # just to test bounds
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'dry_spring_autumn2'),
        lower_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_l,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_l,
            'oxford-irrigated': dry_sp_au_oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_u,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_u,
            'oxford-irrigated': dry_sp_au_oxford_irr_u,

        },

        state_limits={
            9: (['D'], '*', '*'),
            10: (['D'], '*', '*'),
            11: ('*', '*', (0.5, 1)),
            12: ('*', '*', (0.5, 1)),
            1: (['D'], ['H', 'A'], (0.5, 1)),
            2: (['D'], ['H', 'A'], (0.5, 1)),
            3: (['D'], ['H', 'A'], (0.5, 1)),
            4: (['D'], ['H', 'A'], (0.5, 1)),

        },

        n_clusters=5,
        n_pcs=15,
        save_stories=True, correct=True
    )


def dry_spring_autumn3():
    # just to test bounds
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'dry_spring_autumn3'),
        lower_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_l,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_l,
            'oxford-irrigated': dry_sp_au_oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_u,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_u,
            'oxford-irrigated': dry_sp_au_oxford_irr_u,

        },

        state_limits={
            9: (['D'], '*', '*'),
            10: (['D'], '*', '*'),
            11: ('*', '*', (0.5, 1)),
            12: ('*', '*', (0.5, 1)),
            1: (['D', 'A'], ['H'], (0.5, 1)),
            2: (['D', 'A'], ['H'], (0.5, 1)),
            3: (['D', 'A'], ['H'], (0.5, 1)),
            4: (['D', 'A'], ['H'], (0.5, 1)),

        },

        n_clusters=2,
        n_pcs=15,
        save_stories=True, correct=True
    )


def dry_spring_autumn4():
    # just to test bounds
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'dry_spring_autumn4'),
        lower_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_l,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_l,
            'oxford-irrigated': dry_sp_au_oxford_irr_l,
        },
        upper_bound={
            'oxford-dryland': dry_sp_au_oxford_dry_u,
            'eyrewell-irrigated': dry_sp_au_eyrewell_irr_u,
            'oxford-irrigated': dry_sp_au_oxford_irr_u,

        },

        state_limits={
            9: (['D'], '*', '*'),
            10: (['D'], '*', '*'),
            11: ('*', '*', (0.5, 1)),
            12: ('*', '*', (0.5, 1)),
            1: ('*', '*', (0.5, 1)),
            2: (['D', 'A'], ['H', 'A'], (0.5, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.5, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.5, 1)),

        },

        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


def bad_stories_eyrewell():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'bad_stories_eyrewell'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 8 * 1000,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 14 * 1000,
            'oxford-irrigated': None,

        },

        state_limits=None,
        n_clusters=15,
        n_pcs=15,
        save_stories=True, correct=True
    )


def get_autumn_drought2_mon_thresh_add_plot():
    data = pd.read_csv(os.path.join(base_dir, r"autumn_drought2_mon_thresh\prop_pg_cluster_data.csv"))
    out = {}
    for mode, site in default_mode_sites:
        temp = {}
        for m in range(1, 13):
            temp[m] = data.loc[:, f'{site}-{mode}_pg_m{m:02d}'].mean()
        out[(site, mode)] = temp

    return out


def hurt_v1():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'hurt_v1'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 11 * 1000,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 12.5 * 1000,
            'oxford-irrigated': None,

        },

        state_limits={
            2: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True,
        monthly_limits=None,
        plt_additional_line=get_autumn_drought2_mon_thresh_add_plot(),
        plt_additional_label='autumn_drought2_mon_thresh'

    )


def hurt_v2():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'hurt_v2'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 11 * 1000,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 12.5 * 1000,
            'oxford-irrigated': None,

        },

        state_limits={
            9: ('*', ['C', 'A'], '*'),
            10: ('*', ['C', 'A'], '*'),
            2: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True,
        monthly_limits=None,
        plt_additional_line=get_autumn_drought2_mon_thresh_add_plot(),
        plt_additional_label='autumn_drought2_mon_thresh'

    )

def hurt_v3():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'hurt_v3'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 11 * 1000,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 12.5 * 1000,
            'oxford-irrigated': None,

        },

        state_limits={
            9: (['D', 'A'], ['C', 'A'], '*'),
            10: (['D', 'A'], ['C', 'A'], '*'),
            2: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            3: (['D', 'A'], ['H', 'A'], (0.6, 1)),
            4: (['D', 'A'], ['H', 'A'], (0.6, 1)),

        },
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True,
        monthly_limits=None,
        plt_additional_line=get_autumn_drought2_mon_thresh_add_plot(),
        plt_additional_label='autumn_drought2_mon_thresh'

    )





if __name__ == '__main__':
    # autumn_drought_1()
    # autumn_drought_2()
    # dry_summer()
    # all_stories()
    # dry_spring_autumn1()
    # dry_spring_autumn2()
    # dry_spring_autumn3()
    # dry_spring_autumn4()
    # autumn_drought_1_no_tresh()
    # autumn_drought_2_no_thresh()
    # autumn_drought_3_no_thresh()

    # autumn_drought_1_mon_thresh()
    # autumn_drought_2_mon_thresh()
    # autumn_drought_3_mon_thresh()
    bad_stories_eyrewell()
    # hurt_v1()
    # hurt_v2()
    # hurt_v3()
    pass #todo what do I need to run here!!!
