import numpy as np
import math
import pandas as pd
import compas

class Feature_Generator():
    def __init__(self):
        self.diffs = None
        self.diffs_size = 0

        self.wind    = None
        self.speed   = None
        self.X_polar = None
        self.Y_polar = None

    # explicit function to normalize array
    def normalize_2d(self,matrix):
        norm = (matrix-np.max(matrix))/(np.max(matrix)-np.min(matrix))
        norm = np.absolute(norm)
        return norm

    def normalize_1d(self,arr,t_min=0,t_max=1):
        arr = arr.astype('float64')
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        min_val  = min(arr)
        for i in range(0,len(arr)):
            n_value = (((arr[i] - min_val)*diff)/diff_arr) + t_min
            arr[i]  = n_value
        return arr

    def Generate_diffs(self,df):
        self.diffs_size = len(df.LONGITUDE)-1
        self.diffs      = np.zeros(self.diffs_size)

        for i in range(1,self.diffs_size):
            x = abs(df['LONGITUDE'].iloc[i] - df['LONGITUDE'].iloc[i-1])
            y = abs(df['LATITUDE'].iloc[i]  - df['LATITUDE'].iloc[i-1])
            self.diffs[i] = math.sqrt(x**2+y**2)*1000 #norm of differences

        self.diffs = self.normalize_1d(self.diffs,t_min=-1,t_max=1)

    def Generate_wind(self,df):
        self.wind = df.HEADING.to_numpy()
        transform = np.vectorize(compas.winds_to_degree)
        self.wind = transform(self.wind)
        # Get speed data
        self.speed   = df.SPEED.to_numpy()
        self.speed   = self.normalize_1d(self.speed)
        self.X_polar = self.speed*np.cos(self.wind)
        self.Y_polar = self.speed*np.sin(self.wind)