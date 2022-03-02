#### NOTE:

#No libraries imported in this file. The class relies on the following libraries, and should be imported as such:

###import numpy as np
###from sklearn.preprocessing import OneHotEncoder

#####

class logistic_regression:

    #FUNCTION to compute softmax function:
    def softmax(self,dat,weight):
        #compute product of data matrix and weights
        z = -dat@weight

        #use this to compute matrix of probabilities:
        p_matrix = (np.exp(z))/((np.exp(z).sum(axis=1)))[:,None]
        return p_matrix



    #FUNCTION to compute the gradient:
    #NOTE: mu is a hyperparamter for the l2 regularization term added to the loss function.
    #mu is a penalization term for the l2 regularization function
    def gradient(self,data,response,weights,mu):
        #get number of observations in data:
        n = np.shape(data)[0]

        #compute and return gradient:
        p = self.softmax(dat=data, weight=weights)
        grad = (((1/n)*((data.T@response) - (data.T@p)))) + ((2*mu)*weights)
        return grad



    #FUNCTION to perform basic gradient descent:
    #PURPOSE: to estimate the matrix of optimal weights needed to make predictions.
    def descent(self, data, max_iterations, y, num_classes, mu, step_size, epsilon): #HYPERPARAMETER: mu, step_size,epsilon
        #one-hot encode responses:
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.reshape(-1,1))

        #initialize weights:
        weights = np.zeros((np.shape(data)[1], num_classes))
        weights_new = np.zeros((np.shape(data)[1], num_classes))

        #keep track of number of iterations (for troubleshooting purposes):
        i = 0

        #loop to perform gradient descent:
        while i <= max_iterations:
            i += 1

            #calculate gradient:
            derivative = self.gradient(
            data=data,
            response=y,
            weights=weights,
            mu = mu
            )

            #update weights
            weights_new -= step_size*derivative

            #check to see if weights changed significantly or not:
            if np.linalg.norm(weights_new - weights, ord = None) < epsilon or \
            np.linalg.norm(derivative) < epsilon:
                break
            else:
                weights = deepcopy(weights_new)
        return weights



    #FUNCTION to fit model by computing optimal weights using gradient descent:
    def fit(self,data, response, num_classes, max_iterations, mu = 0.01, step_size = 0.01, epsilon = 1e-10):
        return self.descent(data = data, y = response, max_iterations=max_iterations, num_classes = num_classes, mu=mu, step_size=step_size, epsilon = epsilon)



    #FUNCTION to make predictions:
    def predict(self,test_data,weights_from_train):
        #make predictions by multiplying test data by weights from gradient descent:
        p = self.softmax(test_data, weights_from_train)
        return np.argmax(p, axis = 1)
