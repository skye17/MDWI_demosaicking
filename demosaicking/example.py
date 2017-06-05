from utils.image_utils import read_img, save_img
from mdwi.mdwi_demosaicking import MDWI_demosaicking
from postprocessing.postprocessing import postprocessing_step
from metrics.metrics import compute_cpsnr
from timeit import default_timer as timer
import os

input_file = 'data/McM/1.tif'
img_true = read_img(input_file)
results_folder = 'data/McM/results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

## Main algorithm
start = timer()
result = MDWI_demosaicking(input_file, CFA=False)
save_img(result, os.path.join(results_folder, '1_result.tif'))
end = timer()

cpsnr = compute_cpsnr(img_true, result, border=10)
elapsed_time = end - start
print("Main algorithm, time elapsed = %f seconds, cpsnr = %f" % (elapsed_time, cpsnr))

## Postprocessing
start_p = timer()
result_p = postprocessing_step(result)
save_img(result_p, os.path.join(results_folder, '1_result_p.tif'))
end_p = timer()
cpsnr_p = compute_cpsnr(img_true, result_p, border=10)
elapsed_time_p = end_p - start_p
print("Postprocessing, time elapsed = %f seconds, cpsnr = %f" % (elapsed_time_p, cpsnr_p))
