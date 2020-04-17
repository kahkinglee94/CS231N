import numpy as np


def affine_forward(x, w, b):
    """
    :param x: input values where x has shape [N, D]
        - N: minibatch size
        - D: dimension of each example [d_0, d_1, d_2, ... , d_k]
    :param w: weights where w has shape [D, H]
        - H: size of hidden layer
    :param b: bias where bias has shape [H, ]
    :return:
        - out: x * w + b (tuple)
        - cache: x, w, b (tuple)
    """

    dim_size = x[0].shape
    X = x.reshape(x.shape[0], np.prod(dim_size))
    out = X.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    :param dout: Computed gradient from the "right" layer. Has shape [N, M]
    :param cache: x, w, b of current layer
    :return: Gradient with respect to x, w and b. (dx, dw, db)
    """

    x, w, b = cache
    dim_shape = np.prod(x[0].shape)
    batch_num = x.shape[0]
    X = x.reshape(batch_num, dim_shape)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    dw = X.T.dot(dout)
    db = dout.sum(axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    :param x: input from forward propagation
    :return: out, cache
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    :param dout: gradient input
    :param cache:
    :return: output gradient
    """
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    :param x: Data of shape (N, D)
    :param gamma: Scale parameter of shape (D,)
    :param beta: Shift paremeter of shape (D,)
    :param bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      -layernorm: Is layernorm enabled
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    :return:
    A tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    layernorm = bn_param.get('layernorm', 0)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        mu = x.mean(axis=0)
        var = x.var(axis=0) + eps
        std = np.sqrt(var)
        z = (x - mu) / std
        out = gamma * z + beta

        if layernorm == 0:
            # Research learnable parameters
            running_mean = momentum * running_mean + (momentum - 1) * mu
            running_var = momentum * running_var + (momentum - 1) * (std**2)
        cache = {'x':x, 'mean':mu, 'var':var, 'std':std, 'z':z, 'gamma':gamma, 'axis':layernorm}

    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    else:
        raise ValueError('Invalid batchnorm forward mode "$s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def layernorm_forward(x, gamma, beta, ln_param):
    """
    :param x: Input of shape [N, D]
    :param gamma: Scale parameter of shape [D, ]
    :param beta: Shift parameter of shape [D, ]
    :param ln_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
    :return:
    A tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    # using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    ln_param['mode'] = 'train' # same as batch norm in train mode
    ln_param['layernorm'] = 1
    # transpose x, gamma and beta
    out, cache = batchnorm_forward(x, gamma.reshape(-1, 1), beta.reshape(-1, 1), ln_param)
    # transpose output to get original dims
    out = out.T

    return out, cache



def softmax_loss(x, y):
    """
    :param x: input from previous layer. Has shape [N, C]
        - x[i, j] is the score for the jth class of the ith input
        - N: minibatch size
        - C: number of classes
    :param y: labels. Has shape [N, ]
    :return:
        - loss
        - gradient with respect to loss
    """
    # Compute loss
    batch_num = x.shape[0]
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    h = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    k = shifted_logits[np.arange(batch_num), y]
    k = k.reshape(batch_num, 1)
    probs = -k + np.log(h)
    loss = np.sum(probs) / batch_num

    #Compute gradient
    exp_scores = np.exp(shifted_logits)
    dx = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    dx[np.arange(batch_num), y] -= 1
    dx /= batch_num
    return loss, dx