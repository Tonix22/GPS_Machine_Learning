import numpy as np
import math
import pandas as pd
import compas
from scipy.stats import norm
class Feature_Generator():
    def __init__(self):
        self.diffs = None
        self.diffs_size = 0

        self.wind    = None
        self.speed   = None
        self.X_polar = None
        self.Y_polar = None

        self.wfd     = None

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

    def Generate_weight_freq_domain(self,df):
        import datetime
        import time
        time_diffs_size   = len(df.TIME)
        samples_time_diff = np.zeros(time_diffs_size)
        self.wfd          = np.zeros(time_diffs_size)

        for i in range(1,time_diffs_size):
            dt_i = datetime.datetime.strptime(df.TIME[i-1], "%m/%d/%Y %H:%M:%S %p")
            posix_dt_i = time.mktime(dt_i.timetuple())

            dt_f = datetime.datetime.strptime(df.TIME[i], "%m/%d/%Y %H:%M:%S %p")
            posix_dt_f = time.mktime(dt_f.timetuple())

            diff = posix_dt_f-posix_dt_i
            samples_time_diff[i] = diff
        
        #samples_time_diff = self.normalize_1d(samples_time_diff)

        avg_sample_rate = np.max(samples_time_diff)
        self.Generate_diffs(df)
       
        #avg and std model
        mu, std = norm.fit(self.diffs)

        #fit PDF curve
        xmin, xmax = mu-4*std,mu+4*std
        x = np.linspace(xmin, xmax, len(self.diffs))
        normal_dist = norm.pdf(x, mu, std)
        diff_threshold = x[(len(self.diffs)*90)//100]

        for i in range(0,len(samples_time_diff)-1):
            reason = df['REASONS'][i]

            if(self.diffs[i] <= diff_threshold and reason!=7): # reason 7 = device is on
                self.wfd[i]   =  samples_time_diff[i]/avg_sample_rate
            else:
                self.wfd[i] = 1

        import matplotlib.pyplot as plt
        xi = list(range(len(self.wfd)))
        plt.plot(xi,self.wfd, 'b') # Draw blue line
        plt.show()
