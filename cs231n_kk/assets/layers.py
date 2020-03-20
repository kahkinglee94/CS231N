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