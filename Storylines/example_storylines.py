"""
 Author: Matt Hanson
 Created: 24/12/2020 9:20 AM
 """
import pandas as pd
from Storylines.check_storyline import ensure_no_impossible_events
import project_base
import os

# old and depreciated

def build_example_storylines_for_greg(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data = pd.read_excel(project_base.slmmac_dir.joinpath(r"event_definition/example_story_lines.xlsx"),
                         header=[0, 1],
                         skiprows=1
                         )

    ndays = {
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
    rest_vals = {1: 10,
                 2: 17,
                 3: 17,
                 4: 10,
                 5: 7,
                 6: 10,
                 7: 10,
                 8: 10,
                 9: 7,
                 10: 5,
                 11: 5,
                 12: 7,
                 }

    rest_vals = {k: rest_vals[k] / ndays[k] for k in range(1, 13)}

    for k in data.columns.levels[0]:
        storyline = data.loc[:, k].dropna()
        storyline.loc[:, 'month'] = storyline.loc[:, 'month'].astype(int)
        storyline.loc[:, 'year'] = storyline.loc[:, 'year'].astype(int) + 2024
        storyline.loc[storyline.rest > 0, 'rest'] = storyline.loc[:, 'month'].replace(rest_vals)

        try:
            ensure_no_impossible_events(storyline)
        except Exception as val:
            print('{} raised:\n {}'.format(k, val))

        storyline.to_csv(os.path.join(outdir, '{}.csv'.format(k)), index=False)


if __name__ == '__main__':
    # these will no longer run as they do not have the correct index
    build_example_storylines_for_greg(project_base.slmmac_dir.joinpath(r"event_definition/example_storys_for_greg"))
