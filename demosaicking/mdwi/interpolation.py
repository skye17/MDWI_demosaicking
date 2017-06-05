from scipy.ndimage.filters import convolve

from demosaicking.mdwi.gradients import *
from demosaicking.utils.convolution_utils import to_convolve_mat, convolve_h, convolve_v
from demosaicking.utils.image_utils import norm_color


def interpolate_missing_component(CFA, color_mask, color, include_diagonals=True, eps=0.01):
    west_filter = to_convolve_mat(np.array([-0.5, 1., 0.5, 0., 0.]))
    east_filter = to_convolve_mat(np.array([0., 0., 0.5, 1, -0.5]))

    color_w = np.where(color_mask == 0, convolve_h(CFA, west_filter), color)
    color_e = np.where(color_mask == 0, convolve_h(CFA, east_filter), color)
    color_n = np.where(color_mask == 0, convolve_v(CFA, west_filter), color)
    color_s = np.where(color_mask == 0, convolve_v(CFA, east_filter), color)

    grad_n = np.where(color_mask == 0, gradient_north(CFA), color + eps)
    grad_s = np.where(color_mask == 0, gradient_south(CFA), color + eps)
    grad_w = np.where(color_mask == 0, gradient_west(CFA), color + eps)
    grad_e = np.where(color_mask == 0, gradient_east(CFA), color + eps)

    gradients = [grad_n, grad_s, grad_w, grad_e]
    estimates = [color_n, color_s, color_w, color_e]

    if include_diagonals:
        h8 = np.array([-8, 23, -48, 161, 161, -48, 23, -8]) / 256.

        NW_mat = to_convolve_mat(
            np.vstack([np.hstack([np.flip(np.eye(8), 1) * h8, np.zeros((8, 1))]), np.zeros((1, 9))]))
        NE_mat = to_convolve_mat(np.vstack([np.hstack([np.zeros((8, 1)), np.eye(8) * h8]), np.zeros((1, 9))]))
        SW_mat = to_convolve_mat(np.vstack([np.zeros((1, 9)), np.hstack([np.eye(8) * h8, np.zeros((8, 1))])]))
        SE_mat = to_convolve_mat(
            np.vstack([np.zeros((1, 9)), np.hstack([np.zeros((8, 1)), np.flip(np.eye(8), 1) * h8])]))

        color_nw = np.where(color_mask == 0, convolve(CFA, NW_mat), color)
        color_ne = np.where(color_mask == 0, convolve(CFA, NE_mat), color)
        color_sw = np.where(color_mask == 0, convolve(CFA, SW_mat), color)
        color_se = np.where(color_mask == 0, convolve(CFA, SE_mat), color)

        # (7)
        grad_nw = np.where(color_mask == 0, gradient_nw(CFA), color + eps)
        grad_ne = np.where(color_mask == 0, gradient_ne(CFA), color + eps)
        grad_sw = np.where(color_mask == 0, gradient_sw(CFA), color + eps)
        grad_se = np.where(color_mask == 0, gradient_se(CFA), color + eps)

        gradients = gradients + [grad_nw, grad_ne, grad_sw, grad_se]
        estimates = estimates + [color_nw, color_ne, color_sw, color_se]
    inverse_gradients = np.asarray(map(lambda x: 1. / x, gradients))
    estimates = np.asarray(estimates)
    est_sum = np.apply_along_axis(np.sum, 0, estimates * inverse_gradients)
    weights_sum = np.apply_along_axis(np.sum, 0, inverse_gradients)
    color_est = est_sum / weights_sum
    color_est = norm_color(color_est)
    assert (np.sum(np.isnan(color_est)) == 0)
    return color_est


def interpolate_rb_at_br(CFA, G_est, G, g_mask, eps=0.01):
    nw_mat = to_convolve_mat(np.matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    ne_mat = to_convolve_mat(np.matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]]))
    sw_mat = to_convolve_mat(np.matrix([[0, 0, 0], [0, 0, 0], [1, 0, 0]]))
    se_mat = to_convolve_mat(np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]]))

    diff_cfa_g = CFA - G_est
    diff_nw = np.where(g_mask == 0, convolve(diff_cfa_g, nw_mat), G)
    diff_ne = np.where(g_mask == 0, convolve(diff_cfa_g, ne_mat), G)
    diff_sw = np.where(g_mask == 0, convolve(diff_cfa_g, sw_mat), G)
    diff_se = np.where(g_mask == 0, convolve(diff_cfa_g, se_mat), G)

    # (12)
    grad_nw = np.where(g_mask == 0, gradient_rb_nw(CFA, G_est), G + eps)
    grad_ne = np.where(g_mask == 0, gradient_rb_ne(CFA, G_est), G + eps)
    grad_sw = np.where(g_mask == 0, gradient_rb_sw(CFA, G_est), G + eps)
    grad_se = np.where(g_mask == 0, gradient_rb_se(CFA, G_est), G + eps)

    # (13)
    gradients = [grad_nw, grad_ne, grad_sw, grad_se]
    inverse_gradients = np.asarray(map(lambda x: 1. / x, gradients))

    # (14)
    diffs_rb = np.array([diff_nw, diff_ne, diff_sw, diff_se])
    weights_sum = np.apply_along_axis(np.sum, 0, inverse_gradients)

    Z = np.apply_along_axis(np.sum, 0, diffs_rb * inverse_gradients)
    res_z = Z / weights_sum

    rb_mask = np.asarray(1 - g_mask, dtype=bool)

    est_0 = (G_est + res_z) * rb_mask
    est_0 = norm_color(est_0)
    assert (np.sum(np.isnan(est_0)) == 0)
    return est_0


def interpolate_rb_at_g(CFA, R_b_est, B_r_est, r_mask, g_mask, b_mask, eps=0.01):
    rb_mask = np.asarray(1 - g_mask, dtype=bool)
    rg_mask = np.asarray(1 - b_mask, dtype=bool)
    bg_mask = np.asarray(1 - r_mask, dtype=bool)
    r_array = CFA * rg_mask + R_b_est * b_mask
    b_array = CFA * bg_mask + B_r_est * r_mask
    R_g_est = interpolate_missing_component(r_array, rb_mask, CFA, include_diagonals=False, eps=eps) * g_mask
    B_g_est = interpolate_missing_component(b_array, rb_mask, CFA, include_diagonals=False, eps=eps) * g_mask
    assert (np.sum(np.isnan(R_g_est)) == 0)
    assert (np.sum(np.isnan(B_g_est)) == 0)
    return R_g_est, B_g_est
