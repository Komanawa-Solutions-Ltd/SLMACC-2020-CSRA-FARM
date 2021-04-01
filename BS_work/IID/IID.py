"""
Author: Bryn Gibson
This program will calculate the combined probability of a story given probability matrices calculated by
SLAMACC_prob_calc.py
The stories must be provided in the format of a directory containing story files in csv format.
The Column headers for this csv must be year, month, temp_class, precip_class, rest
Stories will be iterated over and total story probability calculated for each story.
Stories probability will be written to a csv file in the same base directory as this file named story_probs.csv
"""
from pathlib import Path
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
from datetime import datetime
import warnings
import os

warnings.simplefilter("ignore")

# for adapting months to format used by BS
months = [None,
          "Jan",
          "Feb",
          "Mar",
          "Apr",
          "May",
          "Jun",
          "Jul",
          "Aug",
          "Sep",
          "Oct",
          "Nov",
          "Dec",
          ]

def run_IID(story_dict, outpath=None, verbose=False, comments=''):
    """

    :param storylines: dictionary of identifier: pd.DataFrame
    :param outdir: none or path, if path save data .csv
    :return:
    """
    base_dir = os.path.dirname(__file__)

    # path for Irrigation CDFS
    ir_path = Path(os.path.join(base_dir, "IrrigationRestriction"))
    ir_files = list(ir_path.glob("ir_CDFs*.csv"))

    assert len(ir_files) == 4, "should be 4 CDFS for calculating restriction probabilities"

    ir_dfs = {x.name.split(".")[0].split("_")[-1]: pd.read_csv(x, comment="#", index_col=0)
              for x in ir_files}

    # base path that transition table files are stored in
    base_path = Path(os.path.join(base_dir, "TransitionProbabilities"))

    # globbing probability files
    trans_files = list(base_path.glob("*_transitions.csv"))
    initial_path = base_path / "initial_matrix.csv"

    # raising error if incomplete probability matrices
    if len(trans_files) != 12 or not initial_path.exists():
        raise FileNotFoundError(f"Error should be 12 trans and 1 initial matrices found "
                                f"{len(trans_files)}, and ini_path exists {initial_path.exists()}")

    # loading probability matrices
    if verbose:
        print("loading probability matrices")
    trans_dfs = {f.name.split("_")[0]: pd.read_csv(f, index_col=0, comment="#") for f in trans_files}
    initial_df = pd.read_csv(initial_path, index_col=0, comment="#")
    assert isinstance(story_dict, dict), 'storylines must be dictionaries'
    for k, v in story_dict.items():
        assert isinstance(v, pd.DataFrame), '{} is not a dataframe'.format(k)

    # dataframe for storing story probabilities
    prob_df = pd.DataFrame()

    # iterating through stories
    for k, v in story_dict.items():

        df = v.sort_values(["year", "month"])  # sorting to ensure
        if verbose:
            print(f"calculating story {k}")

        # simplfying precip states for irrigation restriction probabilities
        df["s_precip"] = df["precip_class"].transform(simplfy_precip)

        df["temp_class"] = df["temp_class"].transform(fix_temp)
        df["precip_class"] = df["precip_class"].transform(fix_pr)

        # calculating class and precip combo for easy reference
        df["class"] = df["temp_class"] + "," + df["precip_class"]
        df["precip_combo"] = None

        for i in range(1, len(df)):
            df["precip_combo"].iloc[i] = df["s_precip"].iloc[i - 1] + "," + df["s_precip"][i]

        # getting initial probability by taking the probability of that month starting in that state
        ini_month_key = months[df["month"].iloc[0]]
        prob = np.log10(initial_df[df["class"].iloc[0]].loc[ini_month_key])  # [year.iloc[0]["class"]].values[0]
        # iterating over story
        for i in range(len(df) - 1):

            month1 = df.iloc[i]
            month2 = df.iloc[i + 1]

            # if month in story has a perscribed restriction value looking up that month / values CDF and interpolating
            # to find probability
            if month1["rest"] != 0:
                if month1["precip_combo"] is not None:
                    ir_cdf_df = ir_dfs[month1["precip_combo"]]
                    y = ir_cdf_df[months[month1["month"]]].values[1:-1]
                    if np.sum(np.isnan(y)) == 0:
                        x = ir_cdf_df.index[1:-1]
                        f = interp1d(x, y)

                        # multiplying the story prob by the irrigation restriction prob
                        prob += np.log10(f(month1["rest"]))
                    else:
                        print(k, "no irrigation restriction CDF for this disaggregation due to lack of data")
                else:
                    print(k, "cannot calculate IR prob for first month as no previous month disaggregation information")
            if trans_dfs[months[month1["month"]]][month1["class"]].loc[month2["class"]] == 0:
                print(k, f"{months[month1['month']]}  {month1['class']} to {month2['class']} zero prob transition")
            prob += np.log10(trans_dfs[months[month1["month"]]][month1["class"]].loc[month2["class"]])

        prob_df = prob_df.append({"ID": k, "log10_prob": prob}, ignore_index=True)

    if outpath is None:
        return prob_df
    else:
        # saving dataframe
        f = open(outpath, "w")
        f.write(f"# This file was generated by IID.py   Author: Bryn Gibson  Date: {datetime.now()}\n")
        f.write('#' + comments + '\n')
        f.write("# Stories with a probability of 0 is likely due to the story having a "
                "combination of prescribed class transitions \n")
        f.write("# that were never experienced in the Weather@home batch 793 dataset "
                "or had a prescribed irrigation restriction value\n")
        f.write("# that was greater than restriction values in restriction_record.csv \n")
        f.write("# Column(A): ID ; story file name \n")
        f.write("# Column(B): log10_prob ; log10 of The product of the probabilities of each transition and "
                "restriction values\n")
        prob_df.to_csv(outpath, mode='a', index=False)
        f.close()

        if verbose:
            print(f"saved to story_probs.csv")
        return prob_df

# methods for reformatting stories to fit our internal formatting
def fix_temp(x):
    if x == "A":
        return "AT"
    else:
        return x


def fix_pr(x):
    if x == "A":
        return "AP"
    else:
        return x


def simplfy_precip(x):
    if x == "D":
        return x
    else:
        return "ND"
if __name__ == '__main__':
    story_dir = Path("./example_storylines")  # default stories to test
    files = story_dir.glob("*.csv")
    story_dict = {x.name.split(".")[0]: pd.read_csv(x, comment="#") for x in files}
    t = run_IID(story_dict, verbose=True, comments='a test releating to example storylines', outpath=None)
    expected = pd.read_csv("story_probs.csv", comment="#")
    print(expected)
    assert (np.isclose(np.log10(expected.prob.values), t.log10_prob.values)).all() #todo this will not work due to new data
    print('passed test')
