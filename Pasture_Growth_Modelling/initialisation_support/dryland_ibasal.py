"""
 Author: Matt Hanson
 Created: 23/11/2020 11:06 AM
 """
import ksl_env

# add basgra nz functions
ksl_env.add_basgra_nz_path()
from supporting_functions.plotting import plot_multiple_results
from check_basgra_python.support_for_tests import establish_org_input, get_lincoln_broadfield, get_woodward_weather, _clean_harvest
from input_output_keys import matrix_weather_keys_pet
from basgra_python import run_basgra_nz

def run_nonirr_lincoln_low_basil(IBASAL):
    params, matrix_weather, days_harvest = establish_org_input('lincoln')

    params['FWCWP'] = 0.40  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW #todo dadb validate dryland
    params['FWCFC'] = 0.80  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW #todo dadb validate dryland
    params['WCST'] = 0.46  # from smap Wakanui_6a.1 plus some manual adj to get 55mm PAW #todo dadb validate dryland
    params['BD'] = 1.22  # from smap Wakanui_6a.1 #todo dadb validate dryland
    params['fixed_removal'] = 0 #todo dadb validate dryland
    params['opt_harvfrin'] = 1 #todo dadb validate dryland
    params['IRRIGF'] = 0 #todo dadb validate dryland

    matrix_weather = get_lincoln_broadfield()
    matrix_weather.loc[:, 'max_irr'] = 10
    matrix_weather.loc[:, 'irr_trig'] = 0
    matrix_weather.loc[:, 'irr_targ'] = 1

    matrix_weather = matrix_weather.loc[:, matrix_weather_keys_pet]

    params['IRRIGF'] = 0  # no irrigation
    params['doy_irr_start'] = 305  # start irrigating in Nov
    params['doy_irr_end'] = 90  # finish at end of march
    params['BASALI'] = IBASAL  # start at 20% basal

    days_harvest = _clean_harvest(days_harvest,matrix_weather)

    out = run_basgra_nz(params, matrix_weather, days_harvest, verbose=False)
    out.loc[:,'per_fc'] = out.loc[:,'WAL']/out.loc[:,'WAFC']
    out.loc[:,'per_paw'] = out.loc[:,'PAW']/out.loc[:,'MXPAW']

    return out

if __name__ == '__main__':
    ibasals = [0,0.1,0.15,.2,0.3]
    data = {
            'IBASAL:{}'.format(e): run_nonirr_lincoln_low_basil(e) for e in ibasals
    }

    plot_multiple_results(data, out_vars=['BASAL', 'DM', 'YIELD','per_paw'])
