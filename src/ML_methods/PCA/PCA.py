import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys, pathlib

dirname  = str(pathlib.Path(__file__).parent.absolute())+"/../../"
sys.path.insert(0, dirname)
from params import PLT_PCA
from params import PLT_COVARIANCE

sys.path.insert(0, dirname+"Data_representation")
import Features_generator


class PCA():
    def __init__(self,X,components,plot=False):
        self.input_data  = X
        self.plot        = plot
        self.components  = components
        self.mat_reduced = self.do_PCA(components)
        #Creating a Pandas DataFrame of reduced Dataset

    def percentage(self,x,total):
        return x / total

    def do_PCA(self,num_components):
        #Subtract the mean of each variable
        X_meaned = self.input_data - np.mean(self.input_data , axis = 0)

        # calculating the covariance matrix of the mean-centered data.
        #rowvar is set to False to get the covariance matrix in the required dimensions.
        cov_mat = np.cov(X_meaned , rowvar = False)
        
        cols = ['Lat', 'Lon', 'Dif', 'Reas', 'X_plr','Y_plr','wfd']

        if(PLT_COVARIANCE == True):

            plt.figure(figsize=(10,10))
            sns.set(font_scale=1.5)
            hm = sns.heatmap(cov_mat, cbar=True, annot=True, square=True,
            fmt='.2f',annot_kws={'size': 12},
            yticklabels=cols,
            xticklabels=cols)
            plt.title('Covariance matrix showing correlation coefficients')
            plt.tight_layout()
            plt.show()

        #The Eigenvectors of the Covariance matrix we get are Orthogonal 
        #to each other and each vector represents a principal axis.

        #Calculating Eigenvalues and Eigenvectors of the covariance matrix
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

        #np.argsort returns an array of indices of the same shape.
        sorted_index = np.argsort(eigen_values)[::-1]
        #sort the eigenvalues in descending order
        sorted_eigenvalue = eigen_values[sorted_index]
        #similarly sort the eigenvectors 
        sorted_eigenvectors = eigen_vectors[:,sorted_index]

        sorted_cols = [None] * len(cols)
        i=0
        for n in sorted_index:
            sorted_cols[i] = cols[n]
            i+=1 
        
        if(self.plot == True):
            
            sum = np.sum(sorted_eigenvalue)
            participation = np.vectorize(self.percentage)
            relevance = participation(sorted_eigenvalue,sum)
            x = np.linspace(0, len(relevance), len(relevance))
            plt.scatter(x,relevance)
            plt.xticks(range(len(sorted_cols)), sorted_cols, size='small')
            plt.title("PCA relevance")
            plt.show()
        
        # select the first n eigenvectors, n is desired dimension
        # of our final reduced data.
        eigenvector_subset = sorted_eigenvectors[:,0:num_components]

        #Transform the data

        #The final dimensions of X_reduced will be ( 20, 2 ) 
        # and originally the data was of higher dimensions ( 20, 5 ).
        #singular value decompisition
        X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()

        return X_reduced


def PCA_analysis(self):
    ftg = Features_generator.Feature_Generator()
    ftg.Generate_diffs(self.filter_by_name)
    ftg.Generate_wind(self.filter_by_name)
    ftg.Generate_weight_freq_domain(self.filter_by_name)
    

    lat = self.filter_by_name["LATITUDE"].to_numpy()[1:]
    lat = lat*1000
    raw = lat - lat[0]
    N = ftg.normalize_1d(raw,t_min=-1,t_max=1)
    X = N

    lon = self.filter_by_name["LONGITUDE"].to_numpy()[1:]
    lon = lon*1000
    raw = lon - lon[0]
    N   = ftg.normalize_1d(raw,t_min=-1,t_max=1)
    X   = np.column_stack((X,N))

    X   = np.column_stack((X,ftg.diffs))

    raw = self.filter_by_name["REASONS"].to_numpy()[1:]
    N   = ftg.normalize_1d(raw,t_min=-1,t_max=1)
    X   = np.column_stack((X,N))
    
    X   = np.column_stack((X,ftg.wfd[1:]))
    X   = np.column_stack((X,ftg.X_polar[1:]))
    X   = np.column_stack((X,ftg.Y_polar[1:]))

    self.PCA = PCA(X,5, plot=False)

    #ftg.normalize_1d(self.PCA.mat_reduced,t_min=0,t_max=1)

    if(PLT_PCA):
        target = self.filter_by_name.iloc[:,1]
        sns.scatterplot(x=self.PCA.mat_reduced[:,0], y=self.PCA.mat_reduced[:,3],s=60,hue = target[1:],palette= 'dark:salmon_r')
        plt.show()



