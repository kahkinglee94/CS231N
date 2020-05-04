from cs231n_kk.assets.layers import *


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_norm_relu_forward(X, W, b, gamma, beta, bn_param, normalization, dropout, do_param):
    bn_cache, do_cache = None, None
    affine_out, affine_cache = affine_forward(X, W, b)
    if normalization == 'batchnorm':
        bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    elif normalization == 'layernorm':
        bn_out, bn_cache = layernorm_forward(affine_out, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(bn_out)
    if dropout:
        do_out, do_cache = dropout_forward(relu_out, do_param)
    return do_out, (affine_cache, bn_cache, relu_cache, do_cache)


def affine_norm_relu_backward(dout, cache, normalization, dropout):
    fc_cache, bn_cache, relu_cache, do_cache = cache
    # dropout
    if dropout:
        dout = dropout_backward(dout, do_cache)