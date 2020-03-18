import numpy as np

def affine_forward(x, w, b):
    # input:
    ## x: input values where x has shape [N, D]
    ### N: minibatch size
    ### D: dimension of each example [d_0, d_1, d_2, ... , d_k]
    ## w: weights where w has shape [D, H]
    ### H: size of hidden layer
    ## d: bias where bias has shape [H, ]

    # output:
    ##out: x * w + b
    ##cache: x, w, b

    dim_size = x[0].shape
    X = x.reshape(x.shape[0], np.prod(dim_size))
    out = X.dot(w) + b
    cache = (x, w, b)
    return out, cache

