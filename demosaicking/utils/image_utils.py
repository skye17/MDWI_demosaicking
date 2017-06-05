import numpy as np
from PIL import Image


def tstack(a):
    a = np.asarray(a)
    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def tsplit(a):
    a = np.asarray(a)
    return np.array([a[..., x] for x in range(a.shape[-1])])


## Reading and writing images
def read_img(fname):
    return np.asarray(Image.open(fname), dtype=np.float32)


def save_img(ar, fname):
    Image.fromarray(ar.round().astype(np.uint8)).save(fname)


def show_img(ar):
    return Image.fromarray(ar.round().astype(np.uint8))


## Norm color range in images
def norm_color(color):
    color = np.where(color < 0, 0., color)
    color = np.where(color > 255, 255., color)
    return color


def norm_img(img):
    R, G, B = tsplit(img)
    R = norm_color(R)
    G = norm_color(G)
    B = norm_color(B)
    return tstack([R, G, B])


def check_rgb(RGB):
    if (np.min(RGB) < 0) or (np.max(RGB) > 255):
        print ("Image is out of range, normalization will be done")
        RGB = norm_img(RGB)
    return RGB
