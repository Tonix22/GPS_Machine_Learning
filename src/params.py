import os
CURRENT_PATH   = os.path.dirname(os.path.abspath(__file__))
FILTERED_FILE  = CURRENT_PATH+"/filtered_outliers.csv"

ORIGINAL_DATA  = CURRENT_PATH+'/../Data/Vehicle_GPS_Data__Department_of_Public_Services.csv'
FILTER_DATA    = CURRENT_PATH+'/filtered_outliers.csv'

THRESHOLD_DIFF = 20
START_DATA  = 0
END_DATA    = -1

PLT_ENABLE     = False
PLT_COVARIANCE = False