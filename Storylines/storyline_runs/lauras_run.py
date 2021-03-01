"""
 Author: Matt Hanson
 Created: 25/02/2021 2:08 PM
 """

import pandas as pd
import ksl_env
import os
import itertools
from copy import deepcopy
from Storylines.storyline_building_support import base_events, map_irrigation
from Storylines.check_storyline import ensure_no_impossible_events
from Climate_Shocks import climate_shocks_env

# todo start here!

default_lauras_story_dir = os.path.join(climate_shocks_env.temp_storyline_dir, 'lauras_run')


def make_storylines(rest_quantiles=[0.75, 0.95], no_irr_event=0.5):
    years = [1, 2, 3]
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasons_m = [(6, 7, 8), (9, 10, 11), (12, 1, 2), (3, 4, 5)]

    # 'Cold', 'Drought', 'Restriction', 'Hot'

    inputdata = pd.read_excel(os.path.join(ksl_env.slmmac_dir, 'storylines\StorylineSpreadsheetWSLB.xlsx'),
                              header=[0, 1, 2], index_col=0).transpose()
    inputdata.columns = inputdata.columns.str.strip()
    for k in inputdata.keys():
        print(k)
        data = pd.DataFrame(index=pd.date_range('2024-07-01', '2027-06-01', freq='MS'),
                            columns=['precip_class', 'temp_class', 'rest'])
        data.index.name = 'date'
        data.loc[:, 'year'] = data.index.year
        data.loc[:, 'month'] = data.index.month
        data.loc[:, 'rest'] = 0
        for i, y, m in data.loc[:, ['year', 'month']].itertuples(True, None):
            t, p, r = base_events[m]
            data.loc[i, 'precip_class'] = p
            data.loc[i, 'temp_class'] = t

        temp = inputdata.loc[:, k]

        for y, (s, sms) in itertools.product(years, zip(seasons, seasons_m)):
            svs = [int(e) for e in temp.loc[y, s, 'Cold'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y

                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'temp_class'] = 'C'

            svs = [int(e) for e in temp.loc[y, s, 'Hot'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue
                    data.loc[idx, 'temp_class'] = 'H'

            svs = [int(e) for e in temp.loc[y, s, 'Drought'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'precip_class'] = 'D'

            svs = [int(e) for e in temp.loc[y, s, 'Wet'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'precip_class'] = 'W'

            svs = [int(e) for e in temp.loc[y, s, 'Restriction'].split(',')]
            for sv, smv in zip(svs, sms):
                if sv == 1:
                    if smv in [7, 8, 9, 10, 11, 12]:
                        use_y = 2023 + y
                    else:
                        use_y = 2024 + y
                    idx = (data.year == use_y) & (data.month == smv)
                    if idx.sum() == 0:
                        continue

                    data.loc[idx, 'rest'] = 1
        pass
        try:
            ensure_no_impossible_events(data)
        except ValueError as v:
            print(f'{k}: \n {v}')

        data.loc[:, 'precip_class_prev'] = data.loc[:, 'precip_class'].shift(1).fillna('A')
        for q in rest_quantiles:
            data_out = deepcopy(data)
            q1 = q - no_irr_event
            for i in data.index:
                data_out.loc[i, 'rest'] = map_irrigation(m=data_out.loc[i, 'month'],
                                                         rest_quantile=q1 * data_out.loc[i, 'rest'] + no_irr_event,
                                                         precip=data_out.loc[i, 'precip_class'],
                                                         prev_precip=data_out.loc[i, 'precip_class_prev'])

                # todo save storyline
            data_out.to_csv(os.path.join(default_lauras_story_dir,
                                         f'{k}-rest-{int(no_irr_event * 100)}-{int(q * 100)}.csv'))


def run_pasture_growth():
    make_storylines()


if __name__ == '__main__':
    make_storylines()
