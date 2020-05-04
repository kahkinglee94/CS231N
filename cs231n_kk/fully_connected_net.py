from cs231n_kk.assets import *
import numpy as np


class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10, dropout=1,
                 normalization=None, reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        :param hidden_dims: A list of integers giving the size of each hidden layer.
        :param input_dim: An integer giving the size of the input.
        :param num_classes: An integer giving the number of classes to classify.
        :param dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        :param normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        :param reg: Scalar giving L2 regularization strength.
        :param weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        :param dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        :param seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        layers_dims = np.hstack([input_dim, hidden_dims, num_classes])
        for i in range(self.num_layers):
            self.params['W'+str(i+1)] = weight_scale * np.random.randn(layers_dims[i], layers_dims[i+1])
            self.params['B'+str(i+1)] = np.zeros(layers_dims[i+1])

        if self.normalization is not None:
            for i in range(self.num_layers - 1):
                self.params['gamma'+str(i+1)] = np.ones(layers_dims[i+1])
                self.params['beta'+str(i+1)] = np.zeros(layers_dims[i+1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode':'train', 'p':dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{'mode':'train'} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        :param X: Input of shape [N, D]
        :param y: Labels
        :return: Same as TwoLayerNet
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.use_dropout['mode'] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        x = X
        caches = []
        gamma, beta, bn_params = None, None, None
        for i in range(self.num_layers - 1):
            w = self.params['W'+str(i+1)]
            b = self.params['B'+str(i+1)]
            if self.normalization is not None:
                gamma = self.params['gamma'+str(i+1)]
                beta = self.params['beta'+str(i+1)]
                bn_params = self.bn_params[i]
            x, cache = affine_norm_relu_forward(x, w, b, gamma, beta, bn_params, self.normalization,
                                                self.use_dropout, self.dropout_param)
            caches.append(cache)
        w = self.params['W'+str(self.num_layers)]
        b = self.params['B'+str(self.num_layers)]
        scores, cache = affine_forward(x, w, b)
        caches.append(cache)

        # if test mode, return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # calculate loss
        loss, softmax_grad = softmax_loss(scores, y)
        for i in range(self.num_layers):
            w = self.params['W'+str(i)]
            loss += 0.5 * self.reg * np.sum(w * w)

        #calculate gradient
        dout = softmax_grad
        dout, dw, db = affine_backward(dout, caches[self.num_layers - 1])
        grads['W'+self.num_layers] = dw + self.reg * self.params['W'+str(self.num_layers)]
        grads['B'+self.num_layers] = db

        for i in range(self.num_layers - 2, -1, -1):
            dx, dw, db, dgamma, dbeta = affine_norm_relu_backward(dout, caches[i], self.normalization, self.use_dropout)

