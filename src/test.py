import sys
import pathlib
from matplotlib.pyplot import scatter
import numpy as np
dirname  = str(pathlib.Path(__file__).parent.absolute()) +"/GPS_data"
sys.path.insert(1, dirname)
import GPS_data
import random
import threading
import multiprocessing
import time

def build_forest(set):
    Y = (set.filter_by_name['ASSET']==153).astype(int).to_numpy()

    wind = set.Feature_Generator("wind",set.filter_by_name)
    wfd  = set.Feature_Generator("WFD",set.filter_by_name)
    X = wind
    X = np.column_stack((X,wfd))
    X = np.column_stack((X,set.filter_by_name['REASONS'].to_numpy()))
    #print("build Forest")
    set.Random_Forest_analsysis(X,Y,20,4)
    #print("Test Forest")

    set.filter_one_sample(153,2) #testing 
    testing_size = len(set.filter_by_name)-1
    it_is = []

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
    set.plot_map()
    return max(it_is)

def random_forest_test():
    set = GPS_data.Data_set_reader()
    set.filter_one_sample(153,0)
    set.filter_one_sample(153,1)
#
    #set.append_frame(320,0)
    #set.append_frame(194,0)
    #set.append_frame(377,0)
    set.append_frame(78,0)

    build_forest(set)
  


def PCA_DBSCAN_test():
    PCA = False
    set = GPS_data.Data_set_reader()
    set.filter_one_sample(153,0)
    #set.append_frame(320,0)
    #set.append_frame(194,0)
    #set.append_frame(377,0)
    #set.append_frame(78,0)
    if(PCA == True):
        set.PCA_analysis()
        set.DBSCAN_analysis(set.PCA.mat_reduced[:,0],set.PCA.mat_reduced[:,1],eps=0.5222222222222223,point=13)

    #other_set = Data_set_reader()
    #other_set.filter_one_sample(153,1)
    #other_set.PCA_analysis()
    #set.DBSCAN.Test_DBSCAN(other_set.PCA.mat_reduced[:,0],other_set.PCA.mat_reduced[:,1])

    lat = set.filter_by_name["LATITUDE"].to_numpy()[1:]
    lon = set.filter_by_name["LONGITUDE"].to_numpy()[1:]
    set.DBSCAN_analysis(lon,lat)

def Better_trainig():
    set = GPS_data.Data_set_reader()
    IDs = set.get_all_IDS()
    print(len(IDs))
    n_ids      = []
    for n in IDs:
        set.filter_one_sample(153,0)
        set.filter_one_sample(153,1)
        set.append_frame(n,0)
        set.PCA_analysis() #set.PCA.mat_reduced[0:,0]
        #
        set.DBSCAN_analysis(set.PCA.mat_reduced[:,[0,1]],set.PCA.mat_reduced[:,2],eps=0.5222222222222223,point=13)
        if(set.DBSCAN.n_clusters_ == 2):
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

#


def Find_good_match(ep_s,ep_end,eps_step,id):
    log_name = "output_"+str(id)+".log"
    print("thread Created "+str(id))
    epsilon = np.linspace(ep_s, ep_end, num=eps_step)
    point   = np.linspace(1, 100, num=100)

    max_score_eps = 0
    max_score_point = 0
    max_score = 0
    score = 0
    for i in range(0,eps_step):
        print("****"*3)
        print("ProcesID: "+ str(id)+" eps: "+str(epsilon[i]))
        print("*****"*3)
        for j in range(10,80):
            score = 0
            set = GPS_data.Data_set_reader()
            IDs = set.get_all_IDS()
            #print(len(IDs))
            n_ids      = []
            for n in IDs:
                set.filter_one_sample(153,0)
                set.append_frame(n,0)
                set.PCA_analysis() #set.PCA.mat_reduced[0:,0]
                set.DBSCAN_analysis(set.PCA.mat_reduced[:,[0,1]],set.PCA.mat_reduced[:,2],eps=epsilon[i],point=point[j])
                if(set.DBSCAN.n_clusters_ == 2):
                    n_ids.append(n)
            IDs = n_ids
            print(IDs, file=open(log_name, 'a'))
            if not IDs:
                print("***"*10, file=open(log_name, 'a'))
                print("No candidates", file=open(log_name, 'a'))
                print("epsilon: "+str(epsilon[i]), file=open(log_name, 'a'))
                print("point: "+str(point[j]), file=open(log_name, 'a'))
                print("***"*10, file=open(log_name, 'a'))
                continue

            set.filter_one_sample(153,0)
            set.append_frame(153,1)

            for n in IDs:
                set.append_frame(n,0)

            score = build_forest(set)

            print("***"*10, file=open(log_name, 'a'))
            print("score: "+str(score), file=open(log_name, 'a'))
            print("epsilon: "+str(epsilon[i]), file=open(log_name, 'a'))
            print("point: "+str(point[j]), file=open(log_name, 'a'))
            print("***"*10, file=open(log_name, 'a'))

            if(score > max_score):
                max_score = score
                max_score_eps   = epsilon[i]
                max_score_point = point[j]

    print("", file=open(log_name, 'a'))
    print("Max score: ",end='', file=open(log_name, 'a'))
    print(max_score, file=open(log_name, 'a'))
    print("Max eps",end='', file=open(log_name, 'a'))
    print(max_score_eps, file=open(log_name, 'a'))

    print("Max point",end='', file=open(log_name, 'a'))
    print(max_score_point,end='', file=open(log_name, 'a'))
    print("", file=open(log_name, 'a'))


#multiprocess evaluation of different scenarios
def find_DBSCAN_hyperparameters():
    x = multiprocessing.Process(target=Find_good_match, args=(0.001,0.1,10,0))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.1,0.2,10,1))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.2,0.3,10,2))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.3,0.4,10,3))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.4,0.5,10,4))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.5,0.6,10,5))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.6,0.7,10,6))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.7,0.8,10,7))
    x.start()

    x = multiprocessing.Process(target=Find_good_match, args=(0.8,0.9,10,8))
    x.start()
    
    x = multiprocessing.Process(target=Find_good_match, args=(0.9,1,10,9))
    x.start()
    