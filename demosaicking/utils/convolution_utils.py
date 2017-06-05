import numpy as np
from scipy.ndimage.filters import convolve1d


def convolve_h(x, y):
    return convolve1d(x, y, mode='mirror', axis=1)


def convolve_v(x, y):
    return convolve1d(x, y, mode='mirror', axis=0)


def to_convolve_mat(mat):
    dims = len(mat.shape)
    res = np.copy(mat)
    for i in xrange(dims):
        res = np.flip(res, i)
    return res
