"""
 Author: Matt Hanson
 Created: 14/04/2021 2:44 PM
 """
from Storylines.storyline_runs.run_random_suite import create_nyr_suite
import time

if __name__ == '__main__':
    # todo re-run with new event data after random suite re-run
    t = time.time()
    #create_nyr_suite(2, True, False)
    #create_nyr_suite(3, True, False)
    create_nyr_suite(5, True, False)
    create_nyr_suite(10, True, False)
    print((time.time() - t) / 60, 'minutes to run 2.5e8 sims')
