import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pathlib,sys

dirname  = str(pathlib.Path(__file__).parent.absolute())+"/../../"
sys.path.insert(0, dirname+"Data_representation")
import Visualize_data

class DB_SCAN():
      plot_DBSCAN = Visualize_data.plot_DBSCAN
      def __init__(self,X,Y):
            self.X   = X
            self.X   = np.column_stack((self.X,Y))
            self.X   = StandardScaler().fit_transform(self.X)

            # Compute DBSCAN
            self.db = DBSCAN(eps=0.7, min_samples=10,algorithm='kd_tree').fit(self.X)

            self.core_samples_mask = np.zeros_like(self.db.labels_, dtype=bool)
            self.core_samples_mask[self.db.core_sample_indices_] = True
            self.labels = self.db.labels_
            # Number of clusters in labels, ignoring noise if present.
            self.n_clusters_ = len(set( self.labels)) - (1 if -1 in  self.labels else 0)
            self.n_noise_    = list( self.labels).count(-1)
            #print('Estimated number of clusters: %d' % self.n_clusters_)
            #print('Estimated number of noise points: %d' % self.n_noise_)
            
      def Test_DBSCAN(self,X,Y):
            self.X   = X
            self.X   = np.column_stack((self.X,Y))
            self.X   = StandardScaler().fit_transform(self.X)
            test     = self.db.fit_predict(self.X)
            print(test)