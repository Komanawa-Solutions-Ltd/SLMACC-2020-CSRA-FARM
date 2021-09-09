"""
 Author: Matt Hanson
 Created: 8/09/2021 12:46 PM
 """
import os
import glob
import ksl_env
import pandas as pd
from Climate_Shocks.climate_shocks_env import temp_storyline_dir
from Storylines.develop_mulit_year_stories import make_multi_year_stories_from_random_suite, \
    create_pg_data_multi_year, run_multi_year_pg_model, plot_multi_year_monthly, plot_multi_year_dif_total, \
    plot_mulit_year_dif_monthly, plot_muli_year_total
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
    if run_pg:
        run_multi_year_pg_model(storyline_dir=new_story_dir, data_dir=new_pg_dir, name=name,
                                desc='test mixing and matching random storylines for 3 years, just for debugging')
    if extract_data:
        create_pg_data_multi_year(storyline_dir=new_story_dir, data_dir=new_pg_dir,
                                  outpath=os.path.join(new_pg_dir, f'{name}-multi_data'))


def test_multi_year2(make_stories=True, run_pg=True, extract_data=True):
    year_stories = {
        0: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\autumn_drought2_mon_"
                     r"thresh\storylines_cluster_000\rsl-*.csv"),
        1: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\autumn_drought2_mon_"
                     r"thresh\storylines_cluster_000\rsl-*.csv"),
        2: glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\autumn_drought2_mon_"
                     r"thresh\storylines_cluster_000\rsl-*.csv"),

    }

    for k, v in year_stories.items():
        print(f'{k}: {len(v)} stories')

    name = 'test_multi_2'
    new_pg_dir = os.path.join(default_pasture_growth_dir, name)
    new_story_dir = os.path.join(temp_storyline_dir, name)

    if make_stories:
        make_multi_year_stories_from_random_suite(outdir=new_story_dir, year_stories=year_stories, n=96, )

    # pasture growth modelling
    if run_pg:
        run_multi_year_pg_model(storyline_dir=new_story_dir, data_dir=new_pg_dir, name=name, nsims_mulit=96 * 3,
                                desc='test mixing and matching random storylines for 3 years, just for debugging')
    if extract_data:
        create_pg_data_multi_year(storyline_dir=new_story_dir, data_dir=new_pg_dir,
                                  outpath=os.path.join(new_pg_dir, f'{name}-multi_data'))


def test_bad_eyrwell(make_stories=True, run_pg=True, extract_data=True):
    temp = glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_"
                     r"stories\bad_stories_eyrewell\storylines_cluster_*\rsl-*.csv")
    year_stories = {
        0: temp,
        1: temp,
        2: temp,

    }

    for k, v in year_stories.items():
        print(f'{k}: {len(v)} stories')

    name = 'bad_eyrewell'
    new_pg_dir = os.path.join(default_pasture_growth_dir, name)
    new_story_dir = os.path.join(temp_storyline_dir, name)

    if make_stories:
        make_multi_year_stories_from_random_suite(outdir=new_story_dir, year_stories=year_stories, n=96, )

    # pasture growth modelling
    if run_pg:
        run_multi_year_pg_model(storyline_dir=new_story_dir, data_dir=new_pg_dir, name=name, nsims_mulit=96,
                                desc='test mixing and matching random storylines for 3 years, just for debugging')
    if extract_data:
        create_pg_data_multi_year(storyline_dir=new_story_dir, data_dir=new_pg_dir,
                                  outpath=os.path.join(new_pg_dir, f'{name}-multi_data'))


def test_hurt_v2(make_stories=True, run_pg=True, extract_data=True, plot=True):
    temp = glob.glob(r"M:\Shared drives\Z2003_SLMACC\outputs_for_ws\norm\possible_final_stories\hurt_v2\storylines"
                     r"_cluster_*\rsl-*.csv")
    year_stories = {
        0: temp,
        1: temp,
        2: temp,

    }

    for k, v in year_stories.items():
        print(f'{k}: {len(v)} stories')

    name = 'hurt_v2'
    new_pg_dir = os.path.join(default_pasture_growth_dir, name)
    new_story_dir = os.path.join(temp_storyline_dir, name)

    if make_stories:
        make_multi_year_stories_from_random_suite(outdir=new_story_dir, year_stories=year_stories, n=96, )

    # pasture growth modelling
    if run_pg:
        run_multi_year_pg_model(storyline_dir=new_story_dir, data_dir=new_pg_dir, name=name, nsims_mulit=96,
                                desc='test mixing and matching random storylines for 3 years, just for debugging')
    if extract_data:
        create_pg_data_multi_year(storyline_dir=new_story_dir, data_dir=new_pg_dir,
                                  outpath=os.path.join(new_pg_dir, f'{name}-multi_data'))

    if plot:  # todo copy to others
        impact_path = os.path.join(new_pg_dir, f'{name}-multi_data.csv')
        out_path = os.path.splitext(impact_path)[0]
        plot_multi_year_monthly(
            outpath=f'{out_path}-month_comp.png',
            mode_sites=default_mode_sites,
            impact_data=pd.read_csv(impact_path),
            nyears=3,
            sup_title=name,
            show=False
        )
        plot_mulit_year_dif_monthly(
            outpath=f'{out_path}-month_dif.png',
            mode_sites=default_mode_sites,
            impact_data=pd.read_csv(impact_path),
            nyears=3,
            sup_title=name,
            show=False
        )
        plot_multi_year_dif_total(
            outpath=f'{out_path}-year_dif.png',
            mode_sites=default_mode_sites,
            impact_data=pd.read_csv(impact_path),
            nyears=3,
            sup_title=name,
            show=False
        )
        plot_muli_year_total(
            outpath=f'{out_path}-year_comp.png',
            mode_sites=default_mode_sites,
            impact_data=pd.read_csv(impact_path),
            nyears=3,
            sup_title=name,
            show=False
        )


if __name__ == '__main__':
    test_hurt_v2(make_stories=False,  # todo re-run plotting function
                 run_pg=False,
                 extract_data=False,
                 plot=True)
