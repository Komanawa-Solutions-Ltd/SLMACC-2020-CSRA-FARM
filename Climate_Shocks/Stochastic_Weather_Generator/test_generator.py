"""
 Author: Matt Hanson
 Created: 5/02/2021 12:21 PM
 """
from Climate_Shocks.get_past_record import get_restriction_record
from Climate_Shocks.Stochastic_Weather_Generator.irrigation_days_generator import MovingBlockBootstrapGenerator

def make_input_data():
    month_len = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }


    org_data = get_restriction_record('detrended', recalc=True)
    data = {}
    for m in range(1, 13):
        temp = org_data.loc[(org_data.month == m) & (org_data.day <=month_len[m])].f_rest.values
        data['m{}'.format(m)] = temp
        assert (len(temp) % month_len[m]) == 0, 'problem with m{}'.format(m)

def create_generator():
    #todo start here and make and test the genertor!
    MovingBlockBootstrapGenerator(input_data, blocktype, block, nsims, data_base_path=None, sim_len=None, nblocksize=None,
                 save_to_nc=True)

if __name__ == '__main__':
    make_input_data()