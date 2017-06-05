from refining import *

from interpolation import *
from demosaicking.utils.bayer_mosaicking import mosaicing_CFA_Bayer, masks_CFA_Bayer
from demosaicking.utils.image_utils import read_img, tstack, check_rgb


def MDWI_demosaicking(input_file, CFA=False, eps=0.01, beta=0.5):
    img_ar = read_img(input_file)
    if not CFA:
        img_shape = img_ar.shape[:2]
        CFA = mosaicing_CFA_Bayer(img_ar)
    else:
        img_shape = img_ar.shape
        CFA = img_ar
    r_mask, g_mask, b_mask = masks_CFA_Bayer(img_shape)
    R = CFA * r_mask
    G = CFA * g_mask
    B = CFA * b_mask
    rb_mask = np.asarray(1 - g_mask, dtype=bool)

    ## Interpolate whole G color plane
    G_est = interpolate_missing_component(CFA, g_mask, G, eps)
    G_est = G * g_mask + G_est * rb_mask

    ## Interpolate R/B at B/R
    RB_est_0 = interpolate_rb_at_br(CFA, G_est, G, g_mask, eps)
    ## Refine R/B at B/R
    B_r_est_refined, R_b_est_refined = refining_step(CFA, RB_est_0, r_mask, b_mask, beta)

    ## Interpolate R/B at G using refined version of R/B at B/R
    R_g_est_1, B_g_est_1 = interpolate_rb_at_g(CFA, R_b_est_refined, B_r_est_refined, r_mask, g_mask, b_mask, eps)

    ## Refine R/B at G using refined estimations of R/B at B/R and R/B at G
    R_g_est_refined_1, B_g_est_refined_1 = refining_step_rb(CFA, R_g_est_1, R_b_est_refined, \
                                                            B_g_est_1, B_r_est_refined, \
                                                            r_mask, g_mask, b_mask, beta)

    R_d = R * r_mask + R_b_est_refined * b_mask + R_g_est_refined_1 * g_mask
    B_d = B * b_mask + B_r_est_refined * r_mask + B_g_est_refined_1 * g_mask
    RGB = tstack([R_d, G_est, B_d])
    RGB = check_rgb(RGB)

    return RGB
