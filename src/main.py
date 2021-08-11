import sys
#include GPS
import os
import pathlib
dirname  = str(pathlib.Path(__file__).parent.absolute()) +"/GPS_data"
sys.path.insert(1, dirname)
from GPS_data import *


def main():
    if not os.path.exists(FILTERED_FILE):
        set = GPS_Noise_removal()
        for n in set.all_ids:
            set.filter_ID(n) # end data is not used
            set.densisty_coord()   # save data with low variance 
    
    set = Data_set_reader()
    set.filter_one_sample(153,0)
    #set.filter_one_sample(153,1)
    #set.append_frame(115,0)
    #set.append_frame(115,1)
    #set.append_frame(78,0)
    #set.append_frame(78,1)
    #set.PCA_analysis()

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
