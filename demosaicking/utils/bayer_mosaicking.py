import numpy as np
from image_utils import tsplit


def masks_CFA_Bayer(shape):
    pattern = 'RGGB'
    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')


def mosaicing_CFA_Bayer(RGB):
    RGB = np.asarray(RGB, dtype=float)

    R, G, B = tsplit(RGB)

    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2])

    CFA = R * R_m + G * G_m + B * B_m

    return CFA
