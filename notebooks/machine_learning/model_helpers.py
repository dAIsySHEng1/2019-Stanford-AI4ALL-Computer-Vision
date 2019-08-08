import numpy as np
from scipy.special import expit as s_curve
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Small helper functions for exponent and dot product
def exp(x): return np.exp(x) 
def dot(x, w): return np.dot(x, w)

def sigmoid(x):
    """
    Given a single value x, return the output of the sigmoid activation function.

    HINT: You can use the help exp(x) function above for e^x
    HINT: The formula for sigmoid can be found on Wikipedia:
          https://en.wikipedia.org/wiki/Sigmoid_function
    """
    ## YOUR CODE HERE
    return exp(x)/(exp(x)+1)
    ## END YOUR CODE

def predict_probability(data, weights):
    """
    Use the data and weights to calculate a probability for each data point.
    For example, if "data" has 100 rows, this function should return an array
    with 100 numbers between 0 and 1.

    HINT: "dot products" can be done with dot(x, w), where w are weights and x are values.
    HINT: Use the sigmoid function which can be called with your sigmoid(x)
    HINT: data is of shape (dataset size, num features), and weights is 
    of shape (num features, 1)
    """
    ## YOUR CODE HERE
    ##    value = dot(row,weights)
    #    log_v = sigmoid(value)
    #    pred.append(log_v)
    ## END YOUR CODE
    #return np.array(pred)[..., None]
    ##[...,None]
    #vector = np.dot(data, weights)
    #i = 0
    #while i < .size:
    #    pred.append(sigmoid(vector[i]))
    #    i += 1
    #return pred
    array = sigmoid(np.dot(data,weights))
    return array[..., None]
    
    
def sgd(data, labels, weights, learning_rate, regularization_rate):
    """
    Loop over all the data and labels, one at a time, and update the weights using the logistic
    regression learning rule.

    HINT: Logistic regression learning update rule is given by: 
    weights += learning_rate*data[i, :]*(labels[i] - prob)
    weights -= regularization_rate*weights

    HINT: You will need "predict_probability" to complete this function.
    HINT: As before, data is of shape (dataset size, num features), and weights is 
    of shape (num features, 1)
    HINT: during each iteration of the loop, you call predict probability, apply the logistic
    regression rule, and then perform the regularization update.
    """
    ## YOUR CODE HERE
    for i in range(len(data)):
        prob = predict_probability(data[i, :], weights)
        weights = weights + learning_rate*data[i, :]*(labels[i] - prob)
        weights = weights - regularization_rate*weights
    ## END YOUR CODE
    return weights

def batch_sgd(data, labels, weights, learning_rate, regularization_rate, batch_size):
    """
    Loop over all the data and labels and update the weights using the logistic
    regression learning rule, averaged over multiple samples.

    HINT: You should use the "create_batches" function below.
    HINT: This function will be very similar to "sgd", but you will need to use
    np.mean(...) to average up multiple gradients.
    """
    data_batch, labels_batch = create_batches(data, labels, batch_size)
    ## YOUR CODE HERE
    #for i in range(len(data_batch)):
    #    prob = predict_probability(data_batch[i], weights)
    #    weights = weights + learning_rate*data_batch[i]*(labels_batch[i] - prob)
    #    weights = weights - regularization_rate*weights
    #    a_weight.append(weights)
    #for j in range(len(a_weight)):
    #    for k in range(len(a_weight)):
    #        mean_lst = []
    #        s_weights = a_weight[k][j]
    #        optim_weight = np.mean(s_weights)
    #        bsgd_w.append(optim_weight)
        
    # do each -> take mean of weight updates -> adjust weights
    for i in range(len(data_batch)):
        labels = labels_batch[i]
        prob = predict_probability(data_batch[i],weights)
        weights = weights + learning_rate*np.mean(data_batch[i]*(labels - prob),axis=0)
        weights = weights - regularization_rate*weights
    return weights
    ## END YOUR CODE
    #return weights
    #return bsgd_w

def create_batches(data, labels, batch_size):
    data_batch = np.array_split(data, len(data)/batch_size)
    labels_batch = np.array_split(labels, len(labels)/batch_size)
    return data_batch, labels_batch

