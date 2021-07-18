"""
 Author: Matt Hanson
 Created: 5/07/2021 5:01 PM
 """
from Storylines.storyline_evaluation.storyline_characteristics_for_impact import storyline_subclusters
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


def good_stories():
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


if __name__ == '__main__':
    autumn_drought_1()
    autumn_drought_2()
    dry_summer()
    # dry_summer2() this one only has 12 stories... causes all sorts of problems
    good_stories()  # todo figure out the stories to take forward!!
