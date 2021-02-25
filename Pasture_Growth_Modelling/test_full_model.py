"""
 Author: Matt Hanson
 Created: 26/02/2021 8:57 AM
 """

from Pasture_Growth_Modelling.full_model_implementation import run_pasture_growth

if __name__ == '__main__':
    run_pasture_growth(
        storyline_path=r'C:\Users\Matt Hanson\python_projects\SLMACC-2020-CSRA\Storylines\storyline_csvs\0-baseline.csv',
        outdir=r"C:\Users\Matt Hanson\Downloads\test_pg_ex_swg", nsims=10, padock_rest=True,
        save_daily=True, description='', swg_dir=r"C:\Users\Matt Hanson\Downloads\example_swg")
