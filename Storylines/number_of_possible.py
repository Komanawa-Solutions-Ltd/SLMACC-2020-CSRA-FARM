"""
 Author: Matt Hanson
 Created: 18/01/2021 12:21 PM
 """
import pandas as pd
import numpy as np
from Storylines.check_storyline import get_acceptable_events
import itertools

if __name__ == '__main__':
    # there are c. 4.7e8 billion possible combinations way too many to actually sample
    # how many do we want to run?

    # number of options:
    # 1     8
    # 2     8
    # 3     8
    # 4     4
    # 5     5
    # 6     3
    # 7     2
    # 8     5
    # 9     8
    # 10    4
    # 11    6
    # 12    8

    rests = [9, 10, 11, 12, 1, 2, 3, 4]
    acceptable = get_acceptable_events()
    out_data = pd.DataFrame(index=pd.Index(range(1, 13), name='month'))
    for k, v in acceptable.items():
        out_data.loc[np.in1d(out_data.index, v), k] = True
    out = out_data.sum(axis=1)
    # out.loc[np.in1d(out.index, rests)] *= 3
    pass
    param = out.loc[[1,2,3,4,5,9,10,11,12]].values
    total_combinations = 1
    for i in param:
        total_combinations = total_combinations * i

    print(total_combinations)
    # print(len(list((itertools.product(*param)))))
