import numpy as np
from numba import jit


@jit
def gradient_north(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(3, shape[0], 1):
        for j in xrange(1, shape[1] - 1, 1):
            result[i][j] = np.abs(CFA[i - 3][j - 1] - CFA[i - 1][j - 1]) + np.abs(CFA[i - 2][j - 1] - CFA[i][j - 1]) + \
                           np.abs(CFA[i - 3][j] - CFA[i - 1][j]) + np.abs(CFA[i - 2][j] - CFA[i][j]) + \
                           np.abs(CFA[i - 3][j + 1] - CFA[i - 1][j + 1]) + np.abs(CFA[i - 2][j + 1] - CFA[i][j + 1])
    return result + eps


@jit
def gradient_south(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(0, shape[0] - 3, 1):
        for j in xrange(1, shape[1] - 1, 1):
            result[i][j] = np.abs(CFA[i][j - 1] - CFA[i + 2][j - 1]) + np.abs(CFA[i + 1][j - 1] - CFA[i + 3][j - 1]) + \
                           np.abs(CFA[i][j] - CFA[i + 2][j]) + np.abs(CFA[i + 1][j] - CFA[i + 3][j]) + \
                           np.abs(CFA[i][j + 1] - CFA[i + 2][j + 1]) + np.abs(CFA[i + 1][j + 1] - CFA[i + 3][j + 1])
    return result + eps


@jit
def gradient_west(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(1, shape[0] - 1, 1):
        for j in xrange(3, shape[1], 1):
            result[i][j] = np.abs(CFA[i - 1][j - 3] - CFA[i - 1][j - 1]) + np.abs(CFA[i - 1][j - 2] - CFA[i - 1][j]) + \
                           np.abs(CFA[i][j - 3] - CFA[i][j - 1]) + np.abs(CFA[i][j - 2] - CFA[i][j]) + \
                           np.abs(CFA[i + 1][j - 3] - CFA[i + 1][j - 1]) + np.abs(CFA[i + 1][j - 2] - CFA[i + 1][j])
    return result + eps


@jit
def gradient_east(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(1, shape[0] - 1, 1):
        for j in xrange(0, shape[1] - 3, 1):
            result[i][j] = np.abs(CFA[i - 1][j] - CFA[i - 1][j + 2]) + np.abs(CFA[i - 1][j + 1] - CFA[i - 1][j + 3]) + \
                           np.abs(CFA[i][j] - CFA[i][j + 2]) + np.abs(CFA[i][j + 1] - CFA[i][j + 3]) + \
                           np.abs(CFA[i + 1][j] - CFA[i + 1][j + 2]) + np.abs(CFA[i + 1][j + 1] - CFA[i + 1][j + 3])
    return result + eps


@jit
def gradient_nw(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(2, shape[0] - 1, 1):
        for j in xrange(2, shape[1] - 1, 1):
            result[i][j] = np.abs(CFA[i - 2][j - 1] - CFA[i - 1][j]) + np.abs(CFA[i][j + 1] - CFA[i - 1][j]) + \
                           np.abs(CFA[i - 1][j - 2] - CFA[i][j - 1]) + np.abs(CFA[i + 1][j] - CFA[i][j - 1]) + \
                           np.abs(CFA[i - 2][j - 2] - CFA[i][j]) + np.abs(CFA[i - 1][j - 1] - CFA[i + 1][j + 1])
    return result + eps


@jit
def gradient_ne(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(2, shape[0] - 1, 1):
        for j in xrange(1, shape[1] - 2, 1):
            result[i][j] = np.abs(CFA[i - 2][j + 1] - CFA[i - 1][j]) + np.abs(CFA[i][j - 1] - CFA[i - 1][j]) + \
                           np.abs(CFA[i - 1][j + 2] - CFA[i][j + 1]) + np.abs(CFA[i + 1][j] - CFA[i][j + 1]) + \
                           np.abs(CFA[i - 2][j + 2] - CFA[i][j]) + np.abs(CFA[i - 1][j + 1] - CFA[i + 1][j - 1])
    return result + eps


@jit
def gradient_sw(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(1, shape[0] - 2, 1):
        for j in xrange(2, shape[1] - 1, 1):
            result[i][j] = np.abs(CFA[i - 1][j] - CFA[i][j - 1]) + np.abs(CFA[i + 1][j - 2] - CFA[i][j - 1]) + \
                           np.abs(CFA[i][j + 1] - CFA[i + 1][j]) + np.abs(CFA[i + 2][j - 1] - CFA[i + 1][j]) + \
                           np.abs(CFA[i + 2][j - 2] - CFA[i][j]) + np.abs(CFA[i + 1][j - 1] - CFA[i - 1][j + 1])
    return result + eps


@jit
def gradient_se(CFA, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(1, shape[0] - 2, 1):
        for j in xrange(1, shape[1] - 2, 1):
            result[i][j] = np.abs(CFA[i - 1][j] - CFA[i][j + 1]) + np.abs(CFA[i + 1][j + 2] - CFA[i][j + 1]) + \
                           np.abs(CFA[i][j - 1] - CFA[i + 1][j]) + np.abs(CFA[i + 2][j + 1] - CFA[i + 1][j]) + \
                           np.abs(CFA[i + 2][j + 2] - CFA[i][j]) + np.abs(CFA[i - 1][j - 1] - CFA[i + 1][j + 1])
    return result + eps


## RB gradients

@jit
def gradient_rb_nw(CFA, G_est, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(2, shape[0] - 1, 1):
        for j in xrange(2, shape[1] - 1, 1):
            result[i][j] = np.abs(CFA[i - 2][j - 1] - CFA[i - 1][j]) + np.abs(CFA[i - 1][j - 2] - CFA[i][j - 1]) + \
                           np.abs(G_est[i - 2][j - 2] - G_est[i - 1][j - 1]) + np.abs(
                G_est[i - 1][j - 1] - G_est[i][j]) + \
                           np.abs(CFA[i - 1][j - 1] - CFA[i + 1][j + 1])
    return result + eps


@jit
def gradient_rb_ne(CFA, G_est, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(2, shape[0] - 1, 1):
        for j in xrange(1, shape[1] - 2, 1):
            result[i][j] = np.abs(CFA[i - 2][j + 1] - CFA[i - 1][j]) + np.abs(CFA[i - 1][j + 2] - CFA[i][j + 1]) + \
                           np.abs(G_est[i - 2][j + 2] - G_est[i - 1][j + 1]) + np.abs(
                G_est[i - 1][j + 1] - G_est[i][j]) + \
                           np.abs(CFA[i - 1][j + 1] - CFA[i + 1][j - 1])
    return result + eps


@jit
def gradient_rb_sw(CFA, G_est, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(1, shape[0] - 2, 1):
        for j in xrange(2, shape[1] - 1, 1):
            result[i][j] = np.abs(CFA[i][j - 1] - CFA[i + 1][j - 2]) + np.abs(CFA[i + 1][j] - CFA[i + 2][j - 1]) + \
                           np.abs(G_est[i + 2][j - 2] - G_est[i + 1][j - 1]) + np.abs(
                G_est[i + 1][j - 1] - G_est[i][j]) + \
                           np.abs(CFA[i - 1][j + 1] - CFA[i + 1][j - 1])
    return result + eps


@jit
def gradient_rb_se(CFA, G_est, eps=0.01):
    shape = CFA.shape
    result = np.copy(CFA)
    for i in xrange(1, shape[0] - 2, 1):
        for j in xrange(1, shape[1] - 2, 1):
            result[i][j] = np.abs(CFA[i][j + 1] - CFA[i + 1][j + 2]) + np.abs(CFA[i + 1][j] - CFA[i + 2][j + 1]) + \
                           np.abs(G_est[i + 2][j + 2] - G_est[i + 1][j + 1]) + np.abs(
                G_est[i + 1][j + 1] - G_est[i][j]) + \
                           np.abs(CFA[i - 1][j - 1] - CFA[i + 1][j + 1])
    return result + eps
