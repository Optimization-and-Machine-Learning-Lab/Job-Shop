from cl_test import test_model
from utils import *

problem_sizes = [[6,6], [10,10], [15,15], [20,15], [20,20], [30,15], [30,20], [50,15], [50,20], [100,20]] 

maxTime = 100
data_mode = 'ta' # ta, dmu
# model_lists = ['uniform_cl', 'classic_cl_20_15']  # 5 Done
model_lists = ['adaptive_cl_15', 'adversarial', 'classic_cl_15_15']  # 2 
# model_lists = ['classic_cl_20_20', 'classic_cl_30_15', 'classic_cl_30_20']    # 3 Done
# model_lists = ['base_15_15', 'base_20_15', 'base_20_20']   # 4 Done
# model_lists = ['base_30_15', 'base_30_20']                             # 1 Done

for model in model_lists:
    
    input_model = model
    output_log = model

    # POMO
    search_mode = 'pomo' # beam, pomo, sampling, greedy, active
    test_model(problem_sizes, maxTime, data_mode, search_mode, input_model, output_log + '_pomo')

    # BEAM
    search_mode = 'beam' # beam, pomo, sampling, greedy, active
    test_model(problem_sizes, maxTime, data_mode, search_mode, input_model, output_log + '_beam')

    # # SAMPLING
    search_mode = 'sampling' # beam, pomo, sampling, greedy, active
    test_model(problem_sizes, maxTime, data_mode, search_mode, input_model, output_log + '_sampling')

    # GREEDY
    search_mode = 'greedy' # beam, pomo, sampling, greedy, active
    test_model(problem_sizes, maxTime, data_mode, search_mode, input_model, output_log + '_greedy')


