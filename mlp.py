from builtins import range
from builtins import object
import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims # dims=[3072,200,200]?
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        #dims[-1] means the last element of the array 'dims'
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers #number of layers use for what?
        layers = self.layers
        ####################################################
        # TODO: Feedforward                      #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        dense1_out=self.layers[0].feedforward(X)
        dense2_out=self.layers[1].feedforward(dense1_out)
        affine_out=self.layers[2].feedforward(dense2_out)
        loss, dx = softmax_loss(affine_out,y)
        ####################################################
        # TODO: Backpropogation                   #
        ####################################################
        affine_dx = self.layers[2].backward(dx)
        dense2_dx = self.layers[1].backward(affine_dx) 
        dense1_dx = self.layers[0].backward(dense2_dx)
        ####################################################
        # TODO: Add L2 regularization               #
        ####################################################
        square_weights=np.sum(layers[0].params[0]**2)+np.sum(layers[1].params[0]**2)+np.sum(layers[2].params[0]**2)
        loss += 0.5*self.reg*square_weights
        
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ####################################################
        # TODO: Use SGD to update variables in layers.    #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        layers=self.layers
        num_layers = self.num_layers
        params=layers[0].params+layers[1].params+layers[2].params
        grads=layers[0].gradients+layers[1].gradients+layers[2].gradients
        
        reg=self.reg
        grads=[grad +reg*params[i] for i, grad in enumerate(grads)]
        #print (params)
        #print (grads)
        for i,param in enumerate(params):
            params[i] -= learning_rate*grads[i]
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #####################################################
        # TODO: Remember to use functions in class       #
        # SoftmaxLayer                          #
        #####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        #relu1_out=np.maximum(X.dot(layers[0].params[0]+layers[0].params[1],0)
        #scores = np.maximum(relu1_out.dot(layers[1].params[0]) + layers[1].params[1], 0).dot(layers[2].params[0]) + layers[2].params[1]
        ## scores is N x C.
         
        #predictions = np.argmax(scores, axis=1)
                             
        for i in range(num_layers):
            X = layers[i].feedforward(X)
        predictions = np.exp(X - np.max(X, axis=1, keepdims=True))
        predictions /= np.sum(predictions, axis=1, keepdims=True)
        predictions = np.argmax(predictions, axis=1)
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc