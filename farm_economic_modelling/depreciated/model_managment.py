import shutil
import os
import traceback
import win32com.client

from ksl_env import slmmac_dir, slmmac_dir_unbacked

remote_model_path = os.path.join(slmmac_dir, 'farm_model', 'base_model.xlsm') # todo adjust for right time/date/parameters
local_model_path = os.path.join(slmmac_dir_unbacked, 'farm_model', 'base_model.xlsm')

if not os.path.exists(remote_model_path):
    raise ValueError('model does not exist in remote')

if not os.path.exists(local_model_path):
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    shutil.copyfile(remote_model_path, local_model_path)


def run_macro(path):
    # DispatchEx is required in the newest versions of Python.
    excel_macro = win32com.client.DispatchEx("Excel.application")
    excel_path = os.path.expanduser(path)
    workbook = excel_macro.Workbooks.Open(Filename=excel_path, ReadOnly=1)
    try:
        print('running macro')
        excel_macro.Application.Run("RunComparison")
        print('macro success')
    except Exception as val:
        print('macro failed')
        traceback.print_exc()
        pass

    # Save the results in case you have generated data
    workbook.Save()
    workbook.Close(False)
    del excel_macro

