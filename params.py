import os
CURRENT_PATH   = os.path.dirname(os.path.abspath(__file__))
FILTERED_FILE  = CURRENT_PATH+"/filtered_outliers.csv"

ORIGINAL_DATA  = CURRENT_PATH+'/Data/Vehicle_GPS_Data__Department_of_Public_Services.csv'
FILTER_PATH    = CURRENT_PATH+'/filtered'
MY_DRIVER      = FILTER_PATH+'/153/FROM_0_TO_126.csv'
DRIVER1        = FILTER_PATH+'/115/FROM_0_TO_114.csv'
DRIVER2        = FILTER_PATH+'/78/FROM_0_TO_181.csv'
VEHICULE_ID    = 153

START_DATA  = 0
END_DATA    = -1
THRESHOLD_DIFF = .020
PLT_ENABLE = False
GENERATE   = False
