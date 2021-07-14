#name: Emilio Tonix Gleason
#python version 3.6
#email: emiliotonix@gmail.com

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplleaflet
import compas
import path_check
import seaborn as sns
import math
from scipy.stats import norm

ORIGINAL_DATA = 'Data/Vehicle_GPS_Data__Department_of_Public_Services.csv'
SELECT_FILTER = 'filtered/40/FROM_0_TO_128.csv'
VEHICULE_ID = 40
START_DATA  = 0
END_DATA    = -1
THRESHOLD_DIFF = .020
PLT_ENABLE = False

class DATA_ANALYSIS():
    def __init__(self):
        #read data csv
        self.df = pd.read_csv(ORIGINAL_DATA)
        self.filter_by_name = None
        self.diff_threshold = THRESHOLD_DIFF
        self.vehicule_id    = VEHICULE_ID
        self.all_ids        = self.df['ASSET'].unique()

    def filter_ID(self,ID,BEGIN,END):
        self.vehicule_id = ID
        self.f1 = self.df['ASSET'] == ID #select a vehicule
        self.df = self.df.sort_values(by=['TIME']) # sort data by time
        print("self.df size: " + str(self.df[self.f1].shape[0]))
        #filter by VEHICULE_ID
        self.filter_by_name = self.df[self.f1][BEGIN:self.df[self.f1].shape[0]]
        #print(self.filter_by_name)

    def save_filter_data(self,name,begin,end):
        #save filter data in csv
        self.filter_by_name.iloc[begin:end].to_csv(name)
        
    def plot_speed_wind(self):
        #Get wind data and transform it to angle
        wind      = self.filter_by_name.HEADING.to_numpy()
        transform = np.vectorize(compas.winds_to_degree)
        wind      = transform(wind)
        # Get speed data
        speed     = self.filter_by_name.SPEED.to_numpy()
        # Plot polar data
        fig = plt.figure()
        ax  = fig.add_subplot(projection='polar')
        c   = ax.scatter(wind, speed, c=wind, cmap='hsv', alpha=0.75)
        plt.show()

    def plot_map(self):
        plt.plot(self.filter_by_name.LONGITUDE, self.filter_by_name.LATITUDE, 'b') # Draw blue line

        for i in range(len(self.filter_by_name.LONGITUDE)-1):
            x = self.filter_by_name['LONGITUDE'].iloc[i]
            y = self.filter_by_name['LATITUDE'].iloc[i]
            plt.plot(x,y, 'rs') # Draw red 
            plt.text(x * (1 + 0.00001), y * (1 + 0.00001) , i, fontsize=12)
            
        mplleaflet.show()
        

    def plot_speed(self):
        xi = list(range(len(self.filter_by_name)))
        plt.plot(xi,self.filter_by_name.SPEED, 'b') # Draw blue line
        plt.show()

    def normal_dist(self,x , mean , sd):
        prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density

    def densisty_coord(self):
        size_of_arr = len(self.filter_by_name.LONGITUDE)-1
        diffs = np.zeros(size_of_arr)
        start = 0
        end   = 0
        normal_dist = None
        self.diff_threshold = THRESHOLD_DIFF

        path_check.create_path(str(self.vehicule_id))
        for i in range(1,size_of_arr):
            x = abs(self.filter_by_name['LONGITUDE'].iloc[i] - self.filter_by_name['LONGITUDE'].iloc[i-1])
            y = abs(self.filter_by_name['LATITUDE'].iloc[i]  - self.filter_by_name['LATITUDE'].iloc[i-1])
            diffs[i] = math.sqrt(x**2+y**2) #norm of differences

            if(diffs[i] > self.diff_threshold):
                #self.filter_by_name.drop(self.filter_by_name.index[i])
                print("new data")
                end   = i
                self.save_filter_data("filtered/{ID}/FROM_{START}_TO_{END}.csv".format(ID = self.vehicule_id,START=start,END=end),start,end)
                start = end
            
            if(i%100 == 0):
                #index range
                b = i-100
                self.diff_threshold = np.max(diffs)
                #avg and std model
                mu, std = norm.fit(diffs[b:i])
                mu = self.diff_threshold
                # Plot the histogram
                if(PLT_ENABLE == True):
                    plt.hist(diffs[b:i], bins=25, density=True, alpha=0.6, color='g')
                #fit PDF curve
                xmin, xmax = mu-4*std,mu+4*std
                x = np.linspace(xmin, xmax, 100)
                normal_dist = norm.pdf(x, mu, std)
                self.diff_threshold = x[95]
                if(PLT_ENABLE == True):
                    print("diff: "+str(x[95]))
                    print("MAX_PDF: "+ str(normal_dist[95]))
                    #Plot the PDF.
                    plt.plot(x, normal_dist, 'k', linewidth=2)
                    title = "Fit results: avg = %.5f,  std = %.5f" % (mu, std)
                    plt.title(title)
                    plt.show()
                

        end = size_of_arr-1
        self.save_filter_data("filtered/{ID}/FROM_{START}_TO_{END}.csv".format(ID = self.vehicule_id,START=start,END=end),start,end)
        if(PLT_ENABLE == True):
            xi = list(range(len(self.filter_by_name)-1))
            plt.plot(xi,diffs, 'b') # Draw blue line
            plt.show()

    def Driving_Reason(self,begin,end):
        reasons = self.df['REASONS'].to_numpy()
        reasons = np.array(reasons, dtype=int)

        counts  = np.bincount(reasons[begin:end])
        max_repeated_value = np.argmax(counts)
        #avg and std model
        mu, std = norm.fit(reasons[begin:end])
        #mu = max_repeated_value
        # Plot the histogram
        plt.hist(reasons[begin:end], bins=25, density=True, alpha=0.6, color='g')
        #fit PDF curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        print("MAX_PDF: "+ str(p[50]))
        # Plot the PDF.
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: avg = %.5f,  std = %.5f" % (mu, std)
        plt.title(title)
        plt.show()

    def heightMap(self):
        #lon = self.filter_by_name.LONGITUDE.to_numpy()
        #lat = self.filter_by_name.LATITUDE.to_numpy()
        #x, y = np.meshgrid(range(lon), range(lat))
        lat = self.filter_by_name.LATITUDE.to_numpy()
        mu, std = norm.fit(lat[0:100])
        xmin, xmax = plt.xlim()
        x_axis = np.linspace(xmin, xmax, 100)
        p_lat = norm.pdf(x_axis, mu, std)

        long = self.filter_by_name.LONGITUDE.to_numpy()
        mu, std = norm.fit(lat[0:100])
        xmin, xmax = plt.xlim()
        x_axis = np.linspace(xmin, xmax, 100)
        p_long = norm.pdf(x_axis, mu, std)


set = DATA_ANALYSIS()

#default
set.filter_ID(VEHICULE_ID,0,100)

#filter all data set
if("generate" in sys.argv):
    for n in set.all_ids:
        set.filter_ID(n,0,100) # end data is not used
        set.densisty_coord()   # save data with low variance 

if("height" in sys.argv):
    set.heightMap()

if("map" in sys.argv):
    set.plot_map()

if("wind" in sys.argv):
    set.plot_speed_wind()

if("diffs" in sys.argv):
    set.densisty_coord()

if("reasons" in sys.argv):
    set.Driving_Reason(0,set.df[set.f1].shape[0])

if("save" in sys.argv):
    set.save_filter_data("filtered/Vehicule_ID_{ID}_FROM_{START}_TO_{END}.csv".format(ID = VEHICULE_ID,START=START_DATA,END=END_DATA),START_DATA,100)