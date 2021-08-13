import sys
import pathlib
from matplotlib.pyplot import scatter
import numpy as np
dirname  = str(pathlib.Path(__file__).parent.absolute()) +"/GPS_data"
sys.path.insert(1, dirname)
import GPS_data

def build_forest(set):
    Y = (set.filter_by_name['ASSET']==153).astype(int).to_numpy()

    wind = set.Feature_Generator("wind",set.filter_by_name)
    wfd  = set.Feature_Generator("WFD",set.filter_by_name)
    X = wind
    X = np.column_stack((X,wfd))
    X = np.column_stack((X,set.filter_by_name['REASONS'].to_numpy()))
    print("build Forest")
    set.Random_Forest_analsysis(X,Y,10,4)
    print("Test Forest")

    it_is = []
    set.filter_one_sample(153,1)
    testing_size = len(set.filter_by_name)-1

    for n in range (10,testing_size):
        wind = set.Feature_Generator("wind",set.filter_by_name.iloc[0:n])
        wfd  = set.Feature_Generator("WFD",set.filter_by_name.iloc[0:n])

        X = wind
        X = np.column_stack((X,wfd))
        X = np.column_stack((X,set.filter_by_name['REASONS'].to_numpy()[0:n]))

        var = set.Forest.predict(X)
        avg_recognized = np.average(var)
        it_is.append(avg_recognized)
        #print("avg_"+str(n)+": "+str(avg_recognized))
    
    set.plot_generic(it_is)
    #set.plot_map()

def random_forest_test():
    set = GPS_data.Data_set_reader()
    set.filter_one_sample(153,0)
    set.append_frame(153,2)

    set.append_frame(115,0)
    #set.append_frame(115,1)
    #set.append_frame(78,0)
    build_forest(set)
  


def PCA_DBSCAN_test():
    set = GPS_data.Data_set_reader()
    set.filter_one_sample(153,0)
    set.append_frame(78,0)
    set.PCA_analysis()
    set.DBSCAN_analysis(set.PCA.mat_reduced[:,0],set.PCA.mat_reduced[:,1])

    #other_set = Data_set_reader()
    #other_set.filter_one_sample(153,1)
    #other_set.PCA_analysis()
    #set.DBSCAN.Test_DBSCAN(other_set.PCA.mat_reduced[:,0],other_set.PCA.mat_reduced[:,1])
    #lat = set.filter_by_name["LATITUDE"].to_numpy()[1:]
    #lon = set.filter_by_name["LONGITUDE"].to_numpy()[1:]
    #set.DBSCAN_analysis(lon,lat)

def Better_trainig():
    set = GPS_data.Data_set_reader()
    IDs = set.get_all_IDS()
    print(len(IDs))
    n_clusters = []
    n_noise    = []
    n_ids      = []
    for n in IDs:
        set.filter_one_sample(153,0)
        set.append_frame(n,0)
        set.PCA_analysis() #set.PCA.mat_reduced[0:,0]
        set.DBSCAN_analysis(set.PCA.mat_reduced[:,[0,1]],set.PCA.mat_reduced[:,2])
        if(set.DBSCAN.n_clusters_ == 2 and set.DBSCAN.n_noise_ < 100):
            #n_clusters.append(set.DBSCAN.n_clusters_)
            #n_noise.append(set.DBSCAN.n_noise_)
            n_ids.append(n)
    IDs = n_ids
    print(IDs)
    
    set.filter_one_sample(153,0)
    set.append_frame(153,1)

    for n in IDs:
        set.append_frame(n,0)
    build_forest(set)
    
    #set.plot_generic(n_clusters,xi = IDs,scatter=True)
    #set.plot_generic(n_noise,   xi = IDs,scatter=True)