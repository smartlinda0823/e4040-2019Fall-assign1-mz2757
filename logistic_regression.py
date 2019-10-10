import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                            #         
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    h = 1 /(1+np.exp(-x))
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    dim = X.shape[1]
    num_train = X.shape[0]
    f_mat = np.zeros_like(W)
    h_mat = np.zeros_like(W)
    loss1 = np.zeros_like(W)
    grad = 0
    y_ = np.zeros([y.shape[0],W.shape[1]])
    for i in range(y.shape[0]):
        y_[i,y[i]] = 1 
    
    for i in range(num_train):
        sample_x = X[i,:]
        for cate in range(W.shape[1]):
            grad = 0
            f_x = 0
            for index in range(dim):
                f_x += W[index,cate]*sample_x[index]
             
            f_mat[i,cate] = f_x
            h_x = sigmoid(f_x)
            loss += y_[i,cate]*np.log(h_x) + (1 - y_[i,cate]) * np.log(1 - h_x)
            grad += (h_x - y_[i,cate]) * sample_x
            h_mat[i,cate] = h_x - y_[i,cate]
            dW[:,cate] = grad.T
            
    loss = (-1 / num_train )* loss + 0.5 * reg * np.sum(W * W)
    dW = 1/ num_train * dW + reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW =np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no    # 
    # explicit loops.                                       #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                       #
    ############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    dim = X.shape[1]
    num_train = X.shape[0]
    f_mat = X.dot(W)
    h_mat = sigmoid(f_mat)
    y_ = np.zeros([y.shape[0],W.shape[1]])
    for i in range(y.shape[0]):
        y_[i,y[i]] = 1
    
    loss = np.sum(y_ * np.log(h_mat) + (1 - y_) * np.log(1 - h_mat))
    loss = -1 / num_train * loss + 0.5 * reg * np.sum(W * W)
    for i in range(num_train):
        sample_x = X[i,:]
        for cate in range(W.shape[1]):
            beta = ((h_mat - y_)[i,cate])
            grad = 0
            grad += beta * sample_x
            dW[:,cate] = grad.T
    dW = 1 / num_train * dW + reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
