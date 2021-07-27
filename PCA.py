import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class PCA():
    def __init__(self,X,components,plot=False):
        self.input_data  = X
        self.plot        = plot
        self.components  = components
        self.mat_reduced = self.do_PCA(components)
        #Creating a Pandas DataFrame of reduced Dataset
        self.principal_df = pd.DataFrame(self.mat_reduced , columns = ['PC1','PC2'])
        if(plot == True):
            plt.figure(figsize = (6,6))
            sns.scatterplot(data = self.principal_df , x = 'PC1',y = 'PC2',palette= 'icefire')
            plt.show()

    def percentage(self,x,total):
        return x / total

    def do_PCA(self,num_components):
        #X = np.random.randint(10,50,100).reshape(20,5)
        #Subtract the mean of each variable
        X_meaned = self.input_data - np.mean(self.input_data , axis = 0)

        # calculating the covariance matrix of the mean-centered data.
        #rowvar is set to False to get the covariance matrix in the required dimensions.
        cov_mat = np.cov(X_meaned , rowvar = False)

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
        
        if(self.plot == True):
            sum = np.sum(sorted_eigenvalue)
            participation = np.vectorize(self.percentage)
            relevance = participation(sorted_eigenvalue,sum)
            x = np.linspace(0, len(relevance), len(relevance))
            plt.scatter(x,relevance)
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

#X = np.random.randint(10,50,100).reshape(20,5)
#mat_reduced = PCA(X,2)


