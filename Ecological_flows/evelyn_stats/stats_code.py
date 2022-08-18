"""
created Evelyn_Charlesworth 
on: 19/08/2022
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# first create a function that allows the data to be read in and stats to be performed
# if necessary, re-code as separate or nested functions later on

def read_and_stats(pathway, file):
    """A function that reads in a file of flows (associated w/ dates) and performs stats on them,
    allowing the outputs to be input into other eqs"""

    flow_df = pd.read_csv(file)