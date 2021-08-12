import numpy as np
import math
class DecisionTree():
    def __init__(self, x, y, n_features, f_idxs,idxs,depth=10, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        #Will make it recursive later
        for i in self.f_idxs: self.find_better_split(i)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth-1, min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth-1, min_leaf=self.min_leaf)

    def std_agg(self,cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)  

    def find_better_split(self, var_idx):
        x, y = self.x[self.idxs,var_idx], self.y[self.idxs]

        sort_idx = np.argsort(x) #index sort
        #sort x and y
        sort_y,sort_x = y[sort_idx], x[sort_idx]

        #initiate right part
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        #initiate left part 
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

        for i in range(0,self.n-self.min_leaf-1):
            #value at that point
            xi,yi = sort_x[i],sort_y[i]
            # split sizes
            lhs_cnt += 1; rhs_cnt -= 1 
            # sum left and sum right
            lhs_sum += yi; rhs_sum -= yi 
            #square sum
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2

            if i<self.min_leaf or xi==sort_x[i+1]:
                continue

            #split score for each iteration is simply the 
            #weighted average of standard deviation of the 
            #two halves with number of rows in each half as their weights.
            #std_agg(cnt, s1, s2) -> sqrt((s2/cnt) - (s1/cnt)**2)
            lhs_std = self.std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = self.std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            #sum of std_i*n_size_split
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt

            #get the shortest score
            if curr_score<self.score:
                #store this column in the variable self.var_idx 
                self.var_idx,self.score,self.split = var_idx,curr_score,xi

    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x[self.idxs,self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf') or self.depth <= 0 
    

    def predict(self, x):
        predict = np.array([self.predict_row(xi) for xi in x])
        return predict

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)