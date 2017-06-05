import numpy as np


def compute_cmse(img_true, img_test):
    shape = img_true.shape[:2]
    cmse = np.sum(np.power(img_true - img_test, 2)) / (3. * shape[0] * shape[1])
    return cmse


def compute_cpsnr(img_true, img_test, border):
    img_true_1 = img_true[border:-border, border:-border]
    img_test_1 = img_test[border:-border, border:-border]
    cmse = compute_cmse(img_true_1, img_test_1)
    cpsnr = 10. * np.log10((255. ** 2) / cmse)
    return cpsnr
