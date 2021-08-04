import sys
from GPS_data import *

def main():
    #if("GPS_noise_removal" in sys.argv):
    if not os.path.isdir(FILTERED_FILE):
        set = GPS_Noise_removal()
        for n in set.all_ids:
            set.filter_ID(n) # end data is not used
            set.densisty_coord()   # save data with low variance 
    
    """
    set.sort_data()


    set.PCA_analysis()

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
    """

if __name__=="__main__":
    main()