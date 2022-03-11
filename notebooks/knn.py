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
            
        accuracy_scratch = sum(test_y == predicted_labels) / len(test_y)
        
        return accuracy_scratch # returns prediction accuracy
        
    def k_tune(train_X,train_y,distance_type,folds):
        ''' The idea is to split training set into l parts. train on l-1 partitions and test on 1 partition.
            Then, find the mean accuracy of all l models for each value k (neighbors). 
            The optimal k will be found for all distance types. 

            input expectations:
                distance_type (string): manhattan, euclidean, or cubic.
                folds (int) 
                train_X (nxp numpy array) (The whole dataset is (m+n) x p)
                test_X (mxp numpy array) 

        '''
        kf = KFold(n_splits = folds, random_state = True, shuffle = True)
        neighbors = [1,2,3,4,5,6]
        accuracy_mat = list()
        mean_accuracy = list()

        for train_index, test_index in kf.split(train_X):       # Iterate through each of 10 folds for each lamda 
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
        return neighbors[max_index], mean_accuracy 

            








