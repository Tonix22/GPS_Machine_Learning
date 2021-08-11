import SingleTree

#n_trees    : Number of uncorrelated trees we ensemble to create the random forest.

#n_features : The number of features to sample and pass onto each tree

#sample_size: The number of rows randomly selected and passed onto each tree. 
#             This is usually equal to total number of rows but can be reduced 
#             to increase performance and decrease correlation of trees in some cases.


class RandomForest():
    def __init__(self, x, y, n_trees, n_features, sample_sz, depth=10, min_leaf=5):
        #just set the seed
        np.random.seed(12)
        #n_features posible clasifications
        #The number of features to sample and pass onto each tree
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        
        print(self.n_features, "sha: ",x.shape[1])

        #outside vars to internal
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf  = x, y, sample_sz, depth, min_leaf
        
        #call function below that generate trees
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        #Randomly permute a sequence, or return a permuted range.
        idxs = np.random.permutation(len(self.y))[:self.sample_sz] # X

        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features] # Y 

        return SingleTree.DecisionTree(self.x.iloc[idxs], self.y[idxs], self.n_features, f_idxs,
                    idxs=np.array(range(self.sample_sz)),depth = self.depth, min_leaf=self.min_leaf)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)