import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplleaflet
import compas
import seaborn as sns
import math
from scipy.stats import norm


VEHICULE_ID = 153
START_DATA  = 0
END_DATA    = 93 
THRESHOLD_DIFF = .020

class DATA_ANALYSIS():
    def __init__(self):
        #read data csv
        self.df = pd.read_csv('Data/Vehicle_GPS_Data__Department_of_Public_Services.csv')
        self.filter_by_name = None

    def filter_ID(self,ID,BEGIN,END):
        self.f1 = self.df['ASSET'] == ID #select a vehicule
        self.df = self.df.sort_values(by=['TIME']) # sort data by time
        print("self.df size: " + str(self.df[self.f1].shape[0]))
        #filter by VEHICULE_ID
        self.filter_by_name = self.df[self.f1][BEGIN:END]
        #print(self.filter_by_name)

    def save_filter_data(self,name):
        #save filter data in csv
        self.filter_by_name.to_csv(name)
        
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
        
        diffs = np.zeros(len(self.filter_by_name.LONGITUDE)-1)
        for i in range(1,len(self.filter_by_name.LONGITUDE)-1):
            x = abs(self.filter_by_name['LONGITUDE'].iloc[i] - self.filter_by_name['LONGITUDE'].iloc[i-1])
            y = abs(self.filter_by_name['LATITUDE'].iloc[i]  - self.filter_by_name['LATITUDE'].iloc[i-1])
            diffs[i] = math.sqrt(x**2+y**2) #norm of differences
            
            if(i%100 == 0 or i == len(self.filter_by_name.LONGITUDE)-2):
                #index range
                b = i-100
                #avg and std model
                mu, std = norm.fit(diffs[b:i])
                # Plot the histogram
                plt.hist(diffs[b:i], bins=25, density=True, alpha=0.6, color='g')
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
            
        xi = list(range(len(self.filter_by_name)-1))
        plt.plot(xi,diffs, 'b') # Draw blue line
        plt.show()

set = DATA_ANALYSIS()

set.filter_ID(VEHICULE_ID,START_DATA,END_DATA)
if("map" in sys.argv):
    set.plot_map()
if("wind" in sys.argv):
    set.plot_speed_wind()
if("diffs" in sys.argv):
    set.densisty_coord()
if("save" in sys.argv):
    set.save_filter_data("filtered/Vehicule_ID_{ID}_FROM_{START}_TO_{END}.csv".format(ID = VEHICULE_ID,START=START_DATA,END=END_DATA))