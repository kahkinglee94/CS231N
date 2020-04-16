from cs231n_kk.assets.layer_util import *
import numpy as np

# get input
# forward propagation
# back propagation


class TwoLayerNet(object):

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        """
        :param input_dim: The dimension size of the input
        :param hidden_dim: An integer giving the size of the hidden layer
        :param num_classes: An integer giving the number of classes to classify
        :param weight_scale: Scalar giving the standard deviation for initialization of the weights
        :param reg: Scalar giving the L2 regularization strength
        :return: None
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["B1"] = np.zeros(hidden_dim)
        self.params["B2"] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data
        :param X: Array of input data of shape (N, d_1, ..., d_k)
        :param y: Array of labels of shape (N,). y[i] gives the label for x[i]
        :return:If y is None, then run a test-time forward pass of the model and return:
                - scores: Array of shape (N, C) giving classification scores, where
                scores[i, c] is the classification score for X[i] and class c.
                If y is not None, then run a training-time forward and backward pass and
                return a tuple of:
                - loss: Scalar value giving the loss
                - grads: Dictionary with the same keys as self.params, mapping parameter
                names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        B1 = self.params["B1"]
        B2 = self.params["B2"]

        X2, relu_cache = affine_relu_forward(X, W1, B1)
        scores, relu2_cache = affine_relu_forward(X2, W2, B2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # calculate loss
        loss, softmax_grad = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  # refer to L2 regularization

        # backpropagate into hidden layer
        dx2, dw2, db2 = affine_relu_backward(softmax_grad, relu2_cache)
        dx, dw, db = affine_relu_backward(dx2, relu_cache)
        grads["W2"] = dw2 + self.reg * W2
        grads["B2"] = db2
        grads["W1"] = dw + self.reg * W1
        grads["B1"] = db

        return loss, grads
