#### NOTE:

#No libraries imported in this file. The class relies on the following libraries, and should be imported as such:

###import numpy as np

class logistic_regression:

    def predict(self,train_X,train_y,test_X,k,distance_type):
        ''' train_X (nxp): design matrix of all training observations 
            train_y (nx1): vehicle class of each training observation. 
            test_X (mxp): design matrix of all testing observations
            k: number of neighbors to consider
            Note: 3 Minkowski distance types are considered: manhattan, euclidean, cubic.'''
        n,p = np.shape(train_X)
        m,p = np.shape(test_X)
        minimum_index = 0
        predicted_labels = list()

        for i in range(m):                     # predicting class of 25 vectors 
            distance = np.zeros(n)             # distance of all 125 training vectors from test vector i
            neighbor_labels = list()
            for j in range(n):                 # finding distance of every training vector from test vector
                if distance_type == 'manhattan':
                    distance[j] = np.linalg.norm(test_X[i,:] - train_X[j,:],ord=1) 
                elif distance_type == 'euclidean':
                    distance[j] = np.linalg.norm(test_X[i,:] - train_X[j,:],ord=2) 
                elif distance_type == 'cubic':
                    distance[j] = np.linalg.norm(test_X[i,:] - train_X[j,:],ord=3) 
            ranked_distance = np.argsort(distance)                               # ranked_distance: indexes of n vectors sorted by increasing distance
            for l in range(k):                                         
                neighbor_labels.append(train_y[ranked_distance[l]])              # finding labels of neighbors
        
            pred_label = max(set(neighbor_labels), key = neighbor_labels.count)  # choosing majority label
            predicted_labels.append(pred_label)                                  # appending majority label of ith observation
        
        return predicted_labels
        
    def k_tune(self,train_X,train_y,test_X,distance_type,folds): # in progress

