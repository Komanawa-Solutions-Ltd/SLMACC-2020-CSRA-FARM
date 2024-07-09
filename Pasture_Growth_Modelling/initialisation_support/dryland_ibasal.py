"""
 Author: Matt Hanson
 Created: 23/11/2020 11:06 AM
 """
from komanawa.basgra_nz_py.supporting_functions.plotting import plot_multiple_results
from komanawa.basgra_nz_py.example_data import establish_org_input, get_lincoln_broadfield, get_woodward_weather, clean_harvest
from komanawa.basgra_nz_py.input_output_keys import matrix_weather_keys_pet
from komanawa.basgra_nz_py.basgra_python import run_basgra_nz

def run_nonirr_lincoln_low_basil(IBASAL):
    params, matrix_weather, days_harvest, doy_irr = establish_org_input('lincoln')

    matrix_weather = get_lincoln_broadfield()
    matrix_weather.loc[:, 'max_irr'] = 10
    matrix_weather.loc[:, 'irr_trig'] = 0
    matrix_weather.loc[:, 'irr_targ'] = 1

    matrix_weather = matrix_weather.loc[:, matrix_weather_keys_pet]

    params['IRRIGF'] = 0  # no irrigation
    params['BASALI'] = IBASAL  # start at 20% basal

    days_harvest = clean_harvest(days_harvest,matrix_weather)

    out = run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False)
    out.loc[:,'per_fc'] = out.loc[:,'WAL']/out.loc[:,'WAFC']
    out.loc[:,'per_paw'] = out.loc[:,'PAW']/out.loc[:,'MXPAW']

    return out

if __name__ == '__main__':
    ibasals = [0,0.1,0.15,.2,0.3]
    data = {
            'IBASAL:{}'.format(e): run_nonirr_lincoln_low_basil(e) for e in ibasals
    }

    plot_multiple_results(data, out_vars=['BASAL', 'DM', 'YIELD','per_paw'])
