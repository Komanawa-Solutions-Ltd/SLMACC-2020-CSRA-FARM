"""
 Author: Matt Hanson
 Created: 5/07/2021 5:01 PM
 """
from Storylines.storyline_evaluation.storyline_characteristics_for_impact import storyline_subclusters
import os

base_dir = r"C:\Users\dumon\Downloads\test_correct"

# bounds are in fraction of new year.
eyrewell_irr_l=0 # todo and propogate into
eyrewell_irr_u=0 # todo

oxford_irr_l=0 # todo
oxford_irr_u=0 # todo

oxford_dry_l=0 # todo
oxford_dry_u=0 # todo

def autumn_drought_1():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought1'),
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


def autumn_drought_2():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'autumn_drought2'),
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


def dry_summer():
    storyline_subclusters(

        outdir=os.path.join(base_dir, 'dry_summer1'),
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

        outdir=os.path.join(base_dir, 'good_stories'),
        lower_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 1,
            'oxford-irrigated': None,
        },
        upper_bound={
            'oxford-dryland': None,
            'eyrewell-irrigated': 30,
            'oxford-irrigated': None,

        },

        state_limits=None,
        n_clusters=10,
        n_pcs=15,
        save_stories=True, correct=True
    )


if __name__ == '__main__':
    good_stories() # todo figure out the stories to take forward!!