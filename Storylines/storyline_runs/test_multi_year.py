"""
 Author: Matt Hanson
 Created: 8/09/2021 12:46 PM
 """
import os
import glob
import ksl_env
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Storylines.develop_mulit_year_stories import make_multi_year_stories_from_random_suite, \
    create_1y_pg_data_multi_year
from Pasture_Growth_Modelling.full_pgr_model_mp import run_full_model_mp, default_pasture_growth_dir, pgm_log_dir, \
    default_mode_sites


def test_multi_year(make_stories=True, run_pg=True, extract_data=True):
    year_stories = {
        0: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\autumn_drought2_mon_"
                     r"thresh\storylines_cluster_000\rsl-*.csv"),
        1: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\good_stor"
                     r"ies\storylines_cluster_001\rsl-*.csv"),
        2: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\autumn_drought2_mon_"
                     r"thresh\storylines_cluster_001\rsl-*.csv"),

    }

    for k, v in year_stories.items():
        print(f'{k}: {len(v)} stories')

    name = 'test_multi'
    new_pg_dir = os.path.join(default_pasture_growth_dir, name)
    new_story_dir = os.path.join(temp_storyline_dir, name)
    if make_stories:
        make_multi_year_stories_from_random_suite(outdir=new_story_dir, year_stories=year_stories, n=16, )

    # pasture growth modelling
    run_stories = glob.glob(os.path.join(new_story_dir, 'mrsl-*.csv'))
    outdirs = [new_pg_dir for e in run_stories]
    if run_pg:
        run_full_model_mp(
            storyline_path_mult=run_stories,
            outdir_mult=outdirs,
            nsims_mult=96,
            log_path=os.path.join(pgm_log_dir, name),
            description_mult='test mixing and matching random storylines for 3 years, just for debugging',
            padock_rest_mult=False,
            save_daily_mult=False,
            verbose=False,
            mode_sites_mult=default_mode_sites,
            re_run=False  # and additional safety

        )

    if extract_data:
        create_1y_pg_data_multi_year(storyline_dir=new_story_dir, data_dir=new_pg_dir,
                                     outpath=os.path.join(new_pg_dir, f'{name}-multi_data'))

def test_multi_year2(make_stories=True, run_pg=True, extract_data=True):
    year_stories = {
        0: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\autumn_drought2_mon_"
                     r"thresh\storylines_cluster_000\rsl-*.csv"),
        2: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\good_stor"
                     r"ies\storylines_cluster_001\rsl-*.csv"),
        1: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\autumn_drought2_mon_"
                     r"thresh\storylines_cluster_001\rsl-*.csv"),

    }

    for k, v in year_stories.items():
        print(f'{k}: {len(v)} stories')

    name = 'test_multi_2'
    new_pg_dir = os.path.join(default_pasture_growth_dir, name)
    new_story_dir = os.path.join(temp_storyline_dir, name)
    if make_stories:
        make_multi_year_stories_from_random_suite(outdir=new_story_dir, year_stories=year_stories, n=16, )

    # pasture growth modelling
    run_stories = glob.glob(os.path.join(new_story_dir, 'mrsl-*.csv'))
    outdirs = [new_pg_dir for e in run_stories]
    if run_pg:
        run_full_model_mp(
            storyline_path_mult=run_stories,
            outdir_mult=outdirs,
            nsims_mult=96,
            log_path=os.path.join(pgm_log_dir, name),
            description_mult='test mixing and matching random storylines for 3 years, just for debugging',
            padock_rest_mult=False,
            save_daily_mult=False,
            verbose=False,
            mode_sites_mult=default_mode_sites,
            re_run=False  # and additional safety

        )

    if extract_data:
        create_1y_pg_data_multi_year(storyline_dir=new_story_dir, data_dir=new_pg_dir,
                                     outpath=os.path.join(new_pg_dir, f'{name}-multi_data'))


# todo test plotting functions
if __name__ == '__main__':
    test_multi_year(make_stories=True,
                    run_pg=False,
                    extract_data=True)
    test_multi_year2(make_stories=True,
                    run_pg=False,
                    extract_data=True)
