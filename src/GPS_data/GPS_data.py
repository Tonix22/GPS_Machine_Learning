#name: Emilio Tonix Gleason
#python version 3.9
#email: emiliotonix@gmail.com

import numpy as np
import pandas as pd
import sys
import os
import pathlib

dirname  = str(pathlib.Path(__file__).parent.absolute())+"/../"
sys.path.insert(0, dirname)
from params import *

sys.path.insert(0, dirname+"ML_methods/PCA")
import PCA

sys.path.insert(0, dirname+"ML_methods/DB_SCAN")
import Db_SCAN

sys.path.insert(0, dirname+"ML_methods/Trees")
import RandomForest

sys.path.insert(0, dirname+"Data_representation")
import Visualize_data
import outlier_by_gausian
import Features_generator

class Data_Set():
    #plotters
    plot_speed_wind = Visualize_data.plot_speed_wind
    plot_map        = Visualize_data.plot_map
    plot_speed      = Visualize_data.plot_speed
    plot_diffs      = Visualize_data.plot_diffs
    plot_Reason     = Visualize_data.plot_Reason
    plot_generic    = Visualize_data.plot_generic

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
    densisty_coord = outlier_by_gausian.densisty_coord
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
    PCA_analysis = PCA.PCA_analysis
    def __init__(self):
        self.PCA    = None
        self.DBSCAN = None
        self.Forest = None
        self.df     = pd.read_csv(FILTER_DATA)
    
    def DBSCAN_analysis(self,X,Y):
        self.DBSCAN = Db_SCAN.DB_SCAN(X,Y)
        self.DBSCAN.plot_DBSCAN()
    
    def Random_Forest_analsysis(self,X,Y,forest_size,n_features):
        self.Forest = RandomForest.RandomForest(X, Y, forest_size, n_features, Y.shape[0])
        

    def Feature_Generator(self,feature,df):
        data = None
        gen     = Features_generator.Feature_Generator()
        if(feature == "diffs"):
            gen.Generate_diffs(df)
            data = gen.diffs

        elif(feature == "wind"):
            gen.Generate_wind(df)
            feature = gen.X_polar
            data = np.column_stack((feature,gen.Y_polar))

        elif(feature == "Frequency"):
            gen.Generate_weight_freq_domain(df)
            data = gen.wfd
        
        return data
        

    def filter_one_sample(self,ID,batch_ID):
        self.vehicule_id = ID
        self.filter_by_name = self.df[self.df['ASSET'] == ID]
        self.filter_by_name = self.filter_by_name[self.filter_by_name['Batch_ID'] == batch_ID]

    def append_frame(self,ID,batch_ID):
        other_vehicule = self.df[self.df['ASSET'] == ID]
        other_vehicule =  other_vehicule[other_vehicule['Batch_ID'] == batch_ID]
        self.filter_by_name = self.filter_by_name.append(other_vehicule, ignore_index=True)
