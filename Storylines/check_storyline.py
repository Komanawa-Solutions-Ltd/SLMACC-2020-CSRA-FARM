"""
 Author: Matt Hanson
 Created: 24/12/2020 9:20 AM
 """
import pandas as pd
import numpy as np

acceptable_events = {  # todo confirm with final sets!
    'temp_class': {
        'C': np.arange(5, 10),
        'A': np.arange(1, 13),
        'H': [11, 12, 1, 2, 3]
    },
    'precip_class': {
        'W': np.arange(5, 10),
        'A': np.arange(1, 13),
        'D': [9, 10, 11, 12, 1, 2, 3, 4, 5]
    },
}


def ensure_no_impossible_events(storyline):
    assert isinstance(storyline, pd.DataFrame)
    assert set(storyline.columns) == {'year', 'month', 'temp_class', 'precip_class', 'rest'}
    assert set(storyline.temp_class.unique()).issubset(['C', 'A', 'H']), 'unexpected classes for temp_class'
    assert set(storyline.precip_class.unique()).issubset(['W', 'A', 'D']), 'unexpected classes for precip_class'
    assert storyline.rest.max() <= 1, 'unexpected values for restrictions'
    assert storyline.rest.min() >= 0, 'unexpected values for restrictions'
    assert set(storyline.month) == set(np.arange(1,13))
    # todo ensure that the months are reasonable and continious

    assert (storyline.loc[np.in1d(storyline.month,[5,6,7,8]),'rest'] == 0).all(), 'irrigation rest in months without irr'

    problems = False
    messages = []
    for k, v in acceptable_events.items():
        for c, amonths in v.items():
            test_data = storyline.loc[storyline[k]==c]
            temp = np.in1d(test_data.month, amonths)
            if not temp.all():
                years = test_data.loc[~temp, 'year'].values
                months = test_data.loc[~temp, 'month'].values
                messages.append('unacceptable class(es):{} in year(s): {} month(s): {}'.format(c, years, months))
                problems = True
    if problems:
        raise ValueError('/n '.join(messages))

