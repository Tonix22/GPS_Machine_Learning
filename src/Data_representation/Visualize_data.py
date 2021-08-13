import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import mplleaflet
from scipy.stats import norm
import Features_generator
#if mplleaflet.show fails check workaround in https://github.com/plotly/plotly.py/issues/2913
#write in python3.9/site-packages/mplleaflet/mplexporter/utils.py line 241 
#if axis._major_tick_kw['gridOn'] and len(gridlines) > 0:

def plot_speed_wind(self):
    ftg = Features_generator.Feature_Generator()
    ftg.Generate_wind(self.filter_by_name)
    # Plot polar data
    fig = plt.figure()
    ax  = fig.add_subplot(projection='polar')
    c   = ax.scatter(ftg.wind, ftg.speed, c=ftg.wind, cmap='hsv', alpha=0.75)
    plt.show()

def plot_map(self):
    plt.plot(self.filter_by_name['LONGITUDE'], self.filter_by_name['LATITUDE'], 'b') # Draw blue line
    for i in range(len(self.filter_by_name.LONGITUDE)-1):
        x = self.filter_by_name['LONGITUDE'].iloc[i]
        y = self.filter_by_name['LATITUDE'].iloc[i]
        plt.plot(x,y, 'rs') # Draw red 

    mplleaflet.show()

def plot_generic(self,arr,xi = None,scatter = False):
    if(xi == None):
        xi = list(range(len(arr)))
    if(scatter):
        plt.scatter(xi, arr)
    else:
        plt.plot(xi,arr,'b') # Draw blue line
    plt.show()

def plot_speed(self):
    xi = list(range(len(self.filter_by_name)))
    plt.plot(xi,self.filter_by_name.SPEED, 'b') # Draw blue line
    plt.show()

def plot_diffs(self):
    ftg = Features_generator.Feature_Generator()
    ftg.Generate_diffs(self.filter_by_name)
    xi = list(range(ftg.diffs_size))
    plt.plot(xi,ftg.diffs, 'b') # Draw blue line
    plt.show()


def plot_Reason(self,begin=0,end=2000):
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

def plot_DBSCAN(self):
    # Black removed and is used for noise instead.
    unique_labels = set(self.labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (self.labels == k)

        xy = self.X[class_member_mask & self.core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = self.X[class_member_mask & ~self.core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % self.n_clusters_)
    #mplleaflet.show()
    plt.show()

