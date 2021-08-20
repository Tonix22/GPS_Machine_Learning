import sys
import numpy as np
#include GPS
import os
import pathlib
dirname  = str(pathlib.Path(__file__).parent.absolute()) +"/GPS_data"
sys.path.insert(1, dirname)
import GPS_data
from params import *
import test

def main():
    if not os.path.exists(FILTERED_FILE):
        set = GPS_data.GPS_Noise_removal()
        for n in set.all_ids:
            set.filter_ID(n) # end data is not used
            set.densisty_coord()   # save data with low variance 
    
    test.random_forest_test()
    #test.PCA_DBSCAN_test()
    #test.Better_trainig()
    #test.Find_good_match()
    #test.find_DBSCAN_hyperparameters()
    

    

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
