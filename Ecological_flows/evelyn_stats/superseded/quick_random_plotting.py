"""
created Evelyn_Charlesworth 
on: 23/08/2022
"""
# Plotting Horizons age tracer data using seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = [["0-10",9.5], ["10-20", 68], ["20-30",58], ["30-50",90], ["50-70",150], ["70-90",136], ["90-110", 134], ["110-150", 152], [ "> 150", 154]]

dataframe = pd.DataFrame(data, columns=["Well depth", 'Mean MRT'])

print(dataframe.head())

sns.catplot(x="Well depth", y="Mean MRT", data=dataframe
            , kind='bar')

plt.show()
