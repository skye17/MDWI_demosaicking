import numpy as np
from scipy.ndimage.filters import convolve
from demosaicking.utils.convolution_utils import to_convolve_mat
from demosaicking.utils.image_utils import norm_color


def refining_step(CFA, RB_est, r_mask, b_mask, beta):
    mu_nw_mat = to_convolve_mat(np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]))
    mu_ne_mat = to_convolve_mat(np.matrix([[0, 0, 1], [0, -1, 0], [0, 0, 0]]))
    mu_sw_mat = to_convolve_mat(np.matrix([[0, 0, 0], [0, -1, 0], [1, 0, 0]]))
    mu_se_mat = to_convolve_mat(np.matrix([[0, 0, 0], [0, -1, 0], [0, 0, 1]]))

    def refine_color(color_mask, other_color_mask):
        mu_ar = CFA * other_color_mask + RB_est * color_mask
        mu_nw = convolve(mu_ar, mu_nw_mat)
        mu_ne = convolve(mu_ar, mu_ne_mat)
        mu_sw = convolve(mu_ar, mu_sw_mat)
        mu_se = convolve(mu_ar, mu_se_mat)
        mus = [mu_nw, mu_ne, mu_sw, mu_se]
        weights = np.asarray(map(lambda x: 1. / (1 + np.abs(x)), mus))
        mus = np.asarray(mus)
        weights_sum = np.apply_along_axis(np.sum, 0, weights)
        est = np.apply_along_axis(np.sum, 0, mus * weights)
        res = (est / weights_sum) * color_mask

        color_est = (beta * RB_est + (1 - beta) * (RB_est + res)) * color_mask
        color_est = norm_color(color_est)
        return color_est

    B_r_est = refine_color(r_mask, b_mask)
    R_b_est = refine_color(b_mask, r_mask)

    assert (np.sum(np.isnan(B_r_est)) == 0)
    assert (np.sum(np.isnan(R_b_est)) == 0)

    return B_r_est, R_b_est


def refining_step_rb(CFA, init_R_g_est, R_b_est, init_B_g_est, B_r_est, r_mask, g_mask, b_mask, beta):
    mu_n_mat = to_convolve_mat(np.matrix([[0, 1, 0], [0, -1, 0], [0, 0, 0]]))
    mu_s_mat = to_convolve_mat(np.matrix([[0, 0, 0], [0, -1, 0], [0, 1, 0]]))
    mu_w_mat = to_convolve_mat(np.matrix([[0, 0, 0], [1, -1, 0], [0, 0, 0]]))
    mu_e_mat = to_convolve_mat(np.matrix([[0, 0, 0], [0, -1, 1], [0, 0, 0]]))

    def refine_color(init_est, color_mask, other_mask, other_est):
        mu_array = CFA * color_mask + init_est * g_mask + other_est * other_mask
        mu_n = convolve(mu_array, mu_n_mat)
        mu_s = convolve(mu_array, mu_s_mat)
        mu_w = convolve(mu_array, mu_w_mat)
        mu_e = convolve(mu_array, mu_e_mat)

        mu = [mu_n, mu_s, mu_w, mu_e]
        weights = np.asarray(map(lambda x: 1. / (1 + np.abs(x)), mu))
        weights_sum = np.apply_along_axis(np.sum, 0, weights)
        mu = np.asarray(mu)
        est = np.apply_along_axis(np.sum, 0, mu * weights)
        res = (est / weights_sum) * g_mask

        color_est = (beta * init_est + (1 - beta) * (init_est + res)) * g_mask
        color_est = norm_color(color_est)
        return color_est

    R_g_est_refined = refine_color(init_R_g_est, r_mask, b_mask, R_b_est)
    B_g_est_refined = refine_color(init_B_g_est, b_mask, r_mask, B_r_est)
    assert (np.sum(np.isnan(R_g_est_refined)) == 0)
    assert (np.sum(np.isnan(B_g_est_refined)) == 0)
    return R_g_est_refined, B_g_est_refined
