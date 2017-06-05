from utils.image_utils import read_img, save_img
from mdwi.mdwi_demosaicking import MDWI_demosaicking
from postprocessing.postprocessing import postprocessing_step
from timeit import default_timer as timer
import os

input_file = 'data/ira/ira2.png'
img_true = read_img(input_file)
results_folder = 'data/ira/results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

## Main algorithm
start = timer()
result = MDWI_demosaicking(input_file, CFA=True)
save_img(result, os.path.join(results_folder, 'ira2_result.tif'))
end = timer()
elapsed_time = end - start
print("Main algorithm, time elapsed = %f seconds" % elapsed_time)

## Postprocessing
start_p = timer()
result_p = postprocessing_step(result)
save_img(result_p, os.path.join(results_folder, 'ira2_result_p.tif'))
end_p = timer()
elapsed_time_p = end_p - start_p
print("Postprocessing, time elapsed = %f seconds" % elapsed_time_p)
