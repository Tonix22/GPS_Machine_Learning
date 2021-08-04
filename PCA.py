import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.stats import norm
import compas

PLT_COVARIANCE = False

class PCA():
    def __init__(self,X,components,plot=False):
        self.input_data  = X
        self.plot        = plot
        self.components  = components
        self.mat_reduced = self.do_PCA(components)
        #Creating a Pandas DataFrame of reduced Dataset
        """
        self.principal_df = pd.DataFrame(self.mat_reduced , columns = ['PC1','PC2'])
        if(plot == True):
            plt.figure(figsize = (6,6))
            sns.scatterplot(data = self.principal_df , x = 'PC1',y = 'PC2',palette= 'icefire')
            plt.show()
        """

    def percentage(self,x,total):
        return x / total

    def do_PCA(self,num_components):
        #X = np.random.randint(10,50,100).reshape(20,5)
        #Subtract the mean of each variable
        X_meaned = self.input_data - np.mean(self.input_data , axis = 0)

        # calculating the covariance matrix of the mean-centered data.
        #rowvar is set to False to get the covariance matrix in the required dimensions.
        cov_mat = np.cov(X_meaned , rowvar = False)
        
        cols = ['Lat', 'Lon', 'Dif', 'Reas', 'X_plr','Y_plr']

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
            print(n)
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

# explicit function to normalize array
    def normalize_2d(self,matrix):
        norm = (matrix-np.max(matrix))/(np.max(matrix)-np.min(matrix))
        norm = np.absolute(norm)
        return norm

def PCA_analysis(self):
    size_of_arr = len(self.filter_by_name.LONGITUDE)-1
    diffs = np.zeros(size_of_arr+1)
    for i in range(1,size_of_arr):
        x = abs(self.filter_by_name['LONGITUDE'].iloc[i] - self.filter_by_name['LONGITUDE'].iloc[i-1])
        y = abs(self.filter_by_name['LATITUDE'].iloc[i]  - self.filter_by_name['LATITUDE'].iloc[i-1])
        diffs[i] = math.sqrt(x**2+y**2) #norm of differences
    
    
    lat = self.filter_by_name["LATITUDE"].to_numpy()
    raw = lat - np.mean(lat, axis = 0)

    N = self.normalize_2d(raw)
    X = N

    lon = self.filter_by_name["LONGITUDE"].to_numpy()
    raw = lat - np.mean(lon, axis = 0)
    N = self.normalize_2d(raw)
    X = np.column_stack((X,N))
    N = self.normalize_2d(diffs)
    X = np.column_stack((X,N))
    
    raw = self.filter_by_name["REASONS"].to_numpy()
    N    = self.normalize_2d(raw)
    X    = np.column_stack((X,N))
    #X = N
    raw  = self.filter_by_name["SPEED"].to_numpy()
    R    = self.normalize_2d(raw)

    #WIND
    wind      = self.filter_by_name.HEADING.to_numpy()
    transform = np.vectorize(compas.winds_to_degree)
    wind      = transform(wind)

    x_polar = R*np.cos(wind)
    y_polar = R*np.sin(wind)
    
    X  = np.column_stack((X,x_polar))
    X  = np.column_stack((X,y_polar))
    
    X = preprocessing.scale(X)

    self.PCA = PCA.PCA(X,2)
    target = self.filter_by_name.iloc[:,1]
    
    sns.scatterplot(x=self.PCA.mat_reduced[:,0], y=self.PCA.mat_reduced[:,1],s=60,hue = target,palette= 'dark:salmon_r')
    plt.show()



