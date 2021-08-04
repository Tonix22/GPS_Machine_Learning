import matplotlib.pyplot as plt
import numpy as np
import compas
from sklearn import preprocessing
import mplleaflet
from scipy.stats import norm
#if mplleaflet.show fails check workaround in https://github.com/plotly/plotly.py/issues/2913
#write in python3.9/site-packages/mplleaflet/mplexporter/utils.py line 241 
#if axis._major_tick_kw['gridOn'] and len(gridlines) > 0:

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
        #plt.text(x * (1 + 0.00001), y * (1 + 0.00001) , i, fontsize=12)
    mplleaflet.show()


def plot_speed(self):
    xi = list(range(len(self.filter_by_name)))
    plt.plot(xi,self.filter_by_name.SPEED, 'b') # Draw blue line
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