import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from Features_generator import Feature_Generator
import os
import sys
dirname  = os.path.dirname(__file__)
filename = os.path.join(dirname, '../')
sys.path.insert(0, filename)

from params import PLT_ENABLE, FILTERED_FILE

def densisty_coord(self):

    Fg = Feature_Generator()
    Fg.Generate_diffs(self.filter_by_name)
    start       = 0
    end         = 0
    normal_dist = None
    batch_size  = 100
    treshhold_idx = (batch_size*95)//100
    batch_id    = 0

    for i in range(1,Fg.diffs_size):
        
        if(Fg.diffs[i] > self.diff_threshold):
            self.filter_by_name.drop(self.filter_by_name.index[i])
            end   = i
            if(end-start >= 50):
                self.save_filter_data(FILTERED_FILE,start,end,batch_id)
                batch_id+=1
            start = end
        
        if(i%batch_size == 0):
            #index range
            b = i-batch_size
            self.diff_threshold = np.max(Fg.diffs[b:i])

            #avg and std model
            mu, std = norm.fit(Fg.diffs[b:i])
            mu = self.diff_threshold

            # Plot the histogram
            if(PLT_ENABLE == True):
                plt.hist(Fg.diffs[b:i], bins=25, density=True, alpha=0.6, color='g')

            #fit PDF curve
            xmin, xmax = mu-4*std,mu+4*std
            x = np.linspace(xmin, xmax, batch_size)
            normal_dist = norm.pdf(x, mu, std)
            self.diff_threshold = x[treshhold_idx]

            if(PLT_ENABLE == True):
                print("diff: "+str(x[treshhold_idx]))
                print("MAX_PDF: "+ str(normal_dist[treshhold_idx]))
                #Plot the PDF.
                plt.plot(x, normal_dist, 'k', linewidth=2)
                title = "Fit results: avg = %.5f,  std = %.5f" % (mu, std)
                plt.title(title)
                plt.show()

            
    end = Fg.diffs_size-1
    if(end-start >= 50):
        self.save_filter_data(FILTERED_FILE,start,end,batch_id)
    if(PLT_ENABLE == True):
        xi = list(range(len(self.filter_by_name)-1))
        plt.plot(xi,Fg.diffs, 'b') # Draw blue line
        plt.show()