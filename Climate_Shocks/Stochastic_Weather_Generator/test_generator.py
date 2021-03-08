"""
 Author: Matt Hanson
 Created: 5/02/2021 12:21 PM
 """
from Climate_Shocks.get_past_record import get_restriction_record
from Climate_Shocks.Stochastic_Weather_Generator.moving_block_bootstrap import MovingBlockBootstrapGenerator

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


    org_data = get_restriction_record('detrended', recalc=False)

    input_data = {}
    sim_len = {}
    for m in range(1, 13):
        temp = org_data.loc[(org_data.month == m) & (org_data.day <=month_len[m])].f_rest.values
        input_data['m{}'.format(m)] = temp
        sim_len['m{}'.format(m)] = month_len[m]
        assert (len(temp) % month_len[m]) == 0, 'problem with m{}'.format(m)
    block = (8, 1.5, 4, 15) #these will not be right
    return input_data, block, sim_len

def create_generator():
    input_data, block, sim_len = make_input_data()
    test_path = r"C:\Users\Matt Hanson\Downloads\test_gen.nc"
    out = MovingBlockBootstrapGenerator(input_data=input_data, blocktype='truncnormal', block=block,
                                        nsims=10000, data_path=test_path, sim_len=sim_len, nblocksize=50,
                                        save_to_nc=True)
    return out

if __name__ == '__main__':

    test = create_generator()
    print('finished making')
    data = test.get_data(1000, key='m1', mean=0.4, tolerance=0.05)
    data2 = test.get_data(1000, key='m1', mean=0.6, tolerance=0.05)

    test.plot_auto_correlation(1000,15, key='m1', show=False)
    test.plot_auto_correlation(1000,15, key='m2', show=False)
    test.plot_means(key='m1')