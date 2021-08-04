import numpy as np
import math
import pandas as pd

class Feature_Generator():
    def __init__(self):
        self.diffs = None
        self.diffs_size = 0

    def Generate_diffs(self,df):
        self.diffs_size = len(df.LONGITUDE)-1
        self.diffs      = np.zeros(self.diffs_size)

        for i in range(1,self.diffs_size):
            x = abs(df['LONGITUDE'].iloc[i] - df['LONGITUDE'].iloc[i-1])
            y = abs(df['LATITUDE'].iloc[i]  - df['LATITUDE'].iloc[i-1])
            self.diffs[i] = math.sqrt(x**2+y**2) #norm of differences