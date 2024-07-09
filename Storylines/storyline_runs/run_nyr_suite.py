"""
 Author: Matt Hanson
 Created: 14/04/2021 2:44 PM
 """

from Storylines.storyline_runs.run_random_suite import create_nyr_suite
import time
import warnings
warnings.warn('this code is not up to date with the current data, see komanawa-slmacc-csra for the most recent version')

if __name__ == '__main__':
    t = time.time()
    create_nyr_suite(1, True, False, correct=False, monthly_data=True)
    # create_nyr_suite(2, True, False, correct=False)
    # create_nyr_suite(3, True, False, correct=False)
    # create_nyr_suite(5, True, False, correct=False)
    # create_nyr_suite(10, True, False, correct=False)
    # create_nyr_suite(2, True, False, correct=True)
    # create_nyr_suite(3, True, False, correct=True)
    # create_nyr_suite(5, True, False, correct=True)
    # create_nyr_suite(10, True, False, correct=True)
    print((time.time() - t) / 60, 'minutes to run 2.5e8 sims')
