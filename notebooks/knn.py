#### NOTE:

#No libraries imported in this file. The class relies on the following libraries, and should be imported as such:

###import numpy as np
###from sklearn.model_selection import KFold

class knn:

    def predict(train_X,train_y,test_X,test_y,k,distance_type):
        '''
        train_X (nxp): design matrix of all training observations 
        train_y (nx1): vehicle class of each training observation. 
        test_X (mxp): design matrix of all testing observations
        k: number of neighbors to consider
        Note: 3 Minkowski distance types are considered: manhattan, euclidean, cubic.
        '''
        n,p = np.shape(train_X)
        m,p = np.shape(test_X)
        minimum_index = 0
        predicted_labels = list()
        dist = dict({'manhattan':1,'euclidean':2,'cubic':3})

        for i in range(m):                     # predicting class of vectors in test set 
            distance = np.zeros(n)             # distance of all training vectors from test vector i
            neighbor_labels = list()
            for j in range(n):                 # finding distance of every training vector from test vector i
                distance[j] = np.linalg.norm(test_X[i,:] - train_X[j,:],ord=dist[distance_type]) 
            ranked_distance = np.argsort(distance)                               # ranked_distance: indices of distance-increasing order
            
            for l in ranked_distance[0:k]:
                neighbor_labels.append(train_y[l]) # finding labels of neighbors

            pred_label = max(set(neighbor_labels), key = neighbor_labels.count)  # choosing majority label
            predicted_labels.append(pred_label)                                  # appending majority label of ith observation

        accuracy_scratch = sum(test_y == predicted_labels) / len(test_y)

        return accuracy_scratch # returns prediction accuracy
        
    def k_tune(self,train_X,train_y,distance_type,folds):
        ''' The idea is to split training set into l parts. train on l-1 partitions and test on 1 partition.
            Then, find the mean accuracy of all l models for each value k (neighbors). 
            The optimal k will be found for all distance types. 

            input expectations:
                distance_type (string): manhattan, euclidean, or cubic.
                folds (int) 
                train_X (nxp numpy array) 
                train_y (nx1 numpy array) 

        '''
        kf = KFold(n_splits = folds, random_state = True, shuffle = True)
        neighbors = [1,3,5,7,9]
        accuracy_mat = list()
        mean_accuracy = list()

        for train_index, test_index in kf.split(train_X):       # Iterate through each of l folds for each lamda 
            temp_X_train = train_X[train_index]
            temp_y_train = train_y[train_index]
            temp_X_test = train_X[test_index]
            temp_y_test = train_y[test_index]

            accuracy_vec = np.zeros(len(neighbors))

            for k in range(len(neighbors)):
                accuracy_vec[k] = predict(temp_X_train,temp_y_train,temp_X_test,temp_y_test,neighbors[k],distance_type)
            accuracy_mat.append(accuracy_vec)
            
        accuracy_mat = np.array(accuracy_mat) 
        folds,neighbor = np.shape(accuracy_mat)
        for i in range(neighbor):
            accuracy = 0
            for j in range(folds):
                accuracy += accuracy_mat[j,i]
            mean_accuracy.append(accuracy / folds)
        max_value = max(mean_accuracy)
        max_index = mean_accuracy.index(max_value)         # contains the mean prediction accuracy for each k neighbors value.
        
        # returned: First object: best k neighbors. Second object: list of prediction accuracies for all values of k
        return neighbors[max_index], max_value


            








