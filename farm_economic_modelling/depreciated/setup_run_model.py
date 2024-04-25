import os
import shutil
import pandas as pd
from openpyxl import load_workbook
from farm_economic_modelling.depreciated.model_managment import run_macro, local_model_path
from ksl_env import slmmac_dir_unbacked

farm_model_run_dir = os.path.join(slmmac_dir_unbacked, 'farm_model', 'runs')
farm_model_result_dir = os.path.join(slmmac_dir_unbacked, 'farm_model', 'results')
os.makedirs(farm_model_run_dir, exist_ok=True)
os.makedirs(farm_model_result_dir, exist_ok=True)


def _check_input_data(input_data):  # todo
    raise NotImplementedError


def setup_run_model(name, input_data):
    temp_model_path = os.path.join(farm_model_run_dir, f'{name}.xlsm')
    if not os.path.exists(temp_model_path):
        shutil.copyfile(local_model_path, temp_model_path)

    # todo check and add the input data!
    _check_input_data(input_data)

    wb = load_workbook(filename=temp_model_path,
                       read_only=False, keep_vba=True)

    sheet = wb['LOScenario1']
    # todo this is the hard point
    # sheet['E22'] = 178 #todo setting range?
    # todo probably better to use .cell format writing: see https://openpyxl.readthedocs.io/en/stable/usage.html
    # todo may need to consider numbers format...
    wb.save(filename=temp_model_path)
    wb.close()

    # run and extract data
    run_macro(temp_model_path)
    data = pd.read_excel(temp_model_path, sheet_name='Results1')  # todo check

    # export data to csv
    data.to_csv(os.path.join(farm_model_result_dir, f'{name}.csv'), index=False)

    # delete temp model
    os.remove(temp_model_path)


# todo make multiprocesing option (and ensure it works)

if __name__ == '__main__':
    test()
