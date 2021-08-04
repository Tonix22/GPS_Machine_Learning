#name: Emilio Tonix Gleason
#python version 3.9
#email: emiliotonix@gmail.com

import numpy as np
import pandas as pd
import PCA
from params import *
from outlier_by_gausian import *

class Data_Set():
    def __init__(self):
        self.df             = None
        self.filter_by_name = None
        self.vehicule_id    = None

    def filter_ID(self,ID,start=0,end=2000):
        self.vehicule_id = ID
        asset = self.df['ASSET'] == ID #select a vehicule
        self.df = self.df.sort_values(by=['TIME']) # sort data by time
        #filter by VEHICULE_ID
        end = min(self.df[asset].shape[0],end)
        self.filter_by_name = self.df[asset][start:end]
        

class GPS_Noise_removal(Data_Set):
    densisty_coord = densisty_coord
    def __init__(self):
        self.df      = pd.read_csv(ORIGINAL_DATA)
        self.all_ids = self.df['ASSET'].unique()
        self.diff_threshold = THRESHOLD_DIFF
        self.fts = True
    
    def save_filter_data(self,name,begin,end,batch_id):
        #save filter data in csv
        id = np.full((end-begin), batch_id)
        self.filter_by_name['Batch_ID'][begin:end] = id

        if(self.fts == True):
            self.filter_by_name.iloc[begin:end,:12].to_csv(name,mode='a')
            self.fts = False
        else:   
            self.filter_by_name.iloc[begin:end,:12].to_csv(name,mode='a', header=False)


class Data_set_reader(Data_Set):
    def __init__(self):
        self.PCA   = None
        self.df    = pd.read_csv(MY_DRIVER)
        self.sort_data()
        #drv1       = pd.read_csv(DRIVER1)
        #drv2       = pd.read_csv(DRIVER2)
        #self.df    = pd.concat([my_driver,drv1,drv2], axis=0)
    
    def sort_data(self):
        self.df = self.df.sort_values(by=['TIME']) # sort data by time
        self.filter_by_name = self.df