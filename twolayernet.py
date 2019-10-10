from __future__ import print_function

import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:
    input -> DenseLayer -> AffineLayer -> softmax loss -> output
    Or more detailed,
    input -> affine transform -> ReLU -> affine transform -> softmax -> output

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_dim=3072, hidden_dim=200, num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.layer1 = DenseLayer(input_dim, hidden_dim, weight_scale=weight_scale)
        self.layer2 = AffineLayer(hidden_dim, num_classes, weight_scale=weight_scale)
        self.reg = reg
        self.velocities = None
        
        self.params = {}
        #self.params['W1'] = 0.001 * np.random.randn(input_dim, hidden_dim)
        #self.params['b1'] = np.zeros(hidden_dim)
        #self.params['W2'] = 0.001 * np.random.randn(hidden_dim, num_classes)
        #self.params['b2'] = np.zeros(num_classes)

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
        ###################################################
        # TODO: Feedforward                     #
        ###################################################
        ###################################################
        #            START OF YOUR CODE          #
        ###################################################  
        out_1 = self.layer1.feedforward(X)
        out_2 = self.layer2.feedforward(out_1)
        loss, dx = softmax_loss(out_2,y)
        ###################################################
        # TODO: Backpropogation,here is just one dense   #
        # layer, it should be pretty easy            #
        ###################################################      
        layer1 = self.layer1
        layer2 = self.layer2
        dx_two = self.layer2.backward(dx)
        dx_three = self.layer1.backward(dx_two)
        ###################################################
        #            END OF YOUR CODE           #
        ###################################################
        # Add L2 regularization
        square_weights = np.sum(layer1.params[0]**2) + np.sum(layer2.params[0]**2)
        loss += 0.5*self.reg*square_weights
        return loss

    def step(self, learning_rate=1e-5, optim='SGD', momentum=0.5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        # creates new lists with all parameters and gradients
        layer1, layer2 = self.layer1, self.layer2 
        params = layer1.params + layer2.params  #self.params = [W, b]
        # W is a [3072,200] random matrix
        #W = weight_scale*np.random.rand(input_dim, output_dim)
        #b = np.zeros(output_dim)
        grads = layer1.gradients + layer2.gradients #self.gradients = [dW, db]
        
        if self.velocities is None:
            self.velocities = [np.zeros_like(param) for param in params]
        
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        ###################################################
        # TODO: Use SGD or SGD with momentum to update   #
        # variables in layer1 and layer2.            #
        ###################################################
        ###################################################
        #            START OF YOUR CODE          #
        ###################################################  
        for i,param in enumerate(params):
            params[i] -= learning_rate*grads[i]
        ###################################################
        #            END OF YOUR CODE           #
        ###################################################
   
        # update parameters in layers
        layer1.update_layer(params[0:2])
        layer2.update_layer(params[2:4])
        
        # Python Reference, actually there is no need for this step.
        # Here add these two lines to make code more clear.
        self.layer1 = layer1
        self.layer2 = layer2


    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        layer1, layer2 = self.layer1, self.layer2
        ###################################################
        # TODO: Remember to use functions in class      #
        # SoftmaxLayer.                        #   
        ###################################################
        ###################################################
        #            START OF YOUR CODE          #
        ###################################################  
        #W1, b1 = self.params['W1'], self.params['b1']
        #W2, b2 = self.params['W2'], self.params['b2']

        scores = np.maximum(X.dot(layer1.params[0]) + layer1.params[1], 0).dot(layer2.params[0]) + layer2.params[1]
        # scores is N x C.
        predictions = np.argmax(scores, axis=1)
        ###################################################
        #            END OF YOUR CODE           #
        ###################################################
        
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
    
    def save_model(self):
        """
        Save model's parameters, including two layer's W and b and reg
        """
        return [self.layer1.params, self.layer2.params, self.reg]
    
    def update_model(self, new_params):
        """
        Update layers and reg with new parameters
        """
        layer1_params, layer2_params, reg = new_params
        
        self.layer1.update_layer(layer1_params)
        self.layer2.update_layer(layer2_params)
        self.reg = reg