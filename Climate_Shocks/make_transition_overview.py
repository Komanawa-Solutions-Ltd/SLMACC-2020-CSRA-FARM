"""
 Author: Matt Hanson
 Created: 24/02/2021 9:13 AM
 """
from Storylines.check_storyline import get_all_zero_prob_transitions

if __name__ == '__main__':
    # run to save zero transition probabilites to help make storylines
    get_all_zero_prob_transitions(save=True)