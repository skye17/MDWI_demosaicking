import numpy as np
from numba import jit


@jit
def gradient_p(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in xrange(1, shape[0] - 1, 1):
        for j in xrange(1, shape[1] - 1, 1):
            result[i][j] = np.abs(ar[i - 1][j - 1] - ar[i + 1][j + 1]) + np.abs(ar[i - 1][j - 1] - ar[i][j]) + \
                           np.abs(ar[i + 1][j + 1] - ar[i][j])
    return result


@jit
def gradient_q(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in xrange(1, shape[0] - 1, 1):
        for j in xrange(1, shape[1] - 1, 1):
            result[i][j] = np.abs(ar[i - 1][j + 1] - ar[i + 1][j - 1]) + np.abs(ar[i - 1][j + 1] - ar[i][j]) + \
                           np.abs(ar[i + 1][j - 1] - ar[i][j])
    return result


@jit
def gradient_h(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in xrange(0, shape[0], 1):
        for j in xrange(1, shape[1] - 1, 1):
            result[i][j] = np.abs(ar[i][j - 1] - ar[i][j + 1]) + np.abs(ar[i][j - 1] - ar[i][j]) + \
                           np.abs(ar[i][j + 1] - ar[i][j])
    return result


@jit
def gradient_v(ar):
    shape = ar.shape
    result = np.copy(ar)
    for i in xrange(1, shape[0] - 1, 1):
        for j in xrange(0, shape[1], 1):
            result[i][j] = np.abs(ar[i - 1][j] - ar[i + 1][j]) + np.abs(ar[i - 1][j] - ar[i][j]) + \
                           np.abs(ar[i + 1][j] - ar[i][j])
    return result
