import sys
import numpy as np
#include GPS
import os
import pathlib
dirname  = str(pathlib.Path(__file__).parent.absolute()) +"/GPS_data"
sys.path.insert(1, dirname)
import GPS_data
from params import *

def main():
    if not os.path.exists(FILTERED_FILE):
        set = GPS_data.GPS_Noise_removal()
        for n in set.all_ids:
            set.filter_ID(n) # end data is not used
            set.densisty_coord()   # save data with low variance 
    
    set = GPS_data.Data_set_reader()
    set.filter_one_sample(153,0)
    set.filter_one_sample(153,1)
    #set.filter_one_sample(153,1)

    set.append_frame(115,0)
    set.append_frame(78,0)
    #set.append_frame(40,0)
    Y = (set.filter_by_name['ASSET']==153).astype(int).to_numpy()
    X = set.Feature_Generator("wind",set.filter_by_name)
    X = np.column_stack((X,set.filter_by_name['REASONS'].to_numpy()))

    set.Random_Forest_analsysis(X,Y,10,3)

    it_is = []
    set.filter_one_sample(153,2)
    testing_size = len(set.filter_by_name)-1
    for n in range (1,testing_size):
        X = set.Feature_Generator("wind",set.filter_by_name[0:n])
        X = np.column_stack((X,set.filter_by_name['REASONS'].to_numpy()[0:n]))
        var = set.Forest.predict(X)
        avg_recognized = np.average(var)
        it_is.append(avg_recognized)
        print("avg_"+str(n)+": "+str(avg_recognized))
    
    set.plot_generic(it_is)
    set.plot_map()
    """
    #set.append_frame(78,1)
    set.PCA_analysis()
    set.DBSCAN_analysis(set.PCA.mat_reduced[:,0],set.PCA.mat_reduced[:,1])
    #other_set = Data_set_reader()
    #other_set.filter_one_sample(153,1)
    #other_set.PCA_analysis()
    #set.DBSCAN.Test_DBSCAN(other_set.PCA.mat_reduced[:,0],other_set.PCA.mat_reduced[:,1])
    #lat = set.filter_by_name["LATITUDE"].to_numpy()[1:]
    #lon = set.filter_by_name["LONGITUDE"].to_numpy()[1:]
    #set.DBSCAN_analysis(lon,lat)
    """

    if("map" in sys.argv):
        set.plot_map()

    if("wind" in sys.argv):
        set.plot_speed_wind()

    if("diffs" in sys.argv):
        set.plot_diffs()

    if("reasons" in sys.argv):
        set.plot_Reason(end = set.filter_by_name.shape[0])

if __name__=="__main__":
    main()
