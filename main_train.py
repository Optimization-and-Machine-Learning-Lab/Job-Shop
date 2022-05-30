from cl_train import train_model

adaptive_threshold = 0.15
train_problem_sizes = [[6,6], [10,10], [15,15], [20,15], [20,20], [30,15], [30,20], [50,15], [50,20], [100,20]] 
maxTime = 100



# # UNIFORM training
# cl_problem_sizes = list(range(0,10)) # indices of problem sizes to choose from train_problem_sizes
# mode = 'uniform' # 'uniform' or 'adaptive' or 'adversarial'
# input_filename = None # give name to continue training from it
# output_filename = 'uniform_cl' # 
# train_model(cl_problem_sizes, train_problem_sizes, maxTime, mode, input_filename, output_filename)



# # INCREMENTAL training
# for i in range(10):
# i = 9
# cl_problem_sizes = [i] # indices of problem sizes to choose from train_problem_sizes
# mode = 'uniform' # 'uniform' or 'adaptive' or 'adversarial'
# input_filename = None # give name to continue training from it
# if i > 0: # increment sizes
#     input_filename = 'classic_cl_%d_%d'%(train_problem_sizes[i-1][0], train_problem_sizes[i-1][1]) # take previous pre-trained model
# output_filename = 'classic_cl_%d_%d'%(train_problem_sizes[i][0], train_problem_sizes[i][1]) # 'classic_cl_6_6'
# train_model(cl_problem_sizes, train_problem_sizes, maxTime, mode, input_filename, output_filename, classic=True)

# # BASE training
i = 6
cl_problem_sizes = [i] # indices of problem sizes to choose from train_problem_sizes
mode = 'uniform' # 'uniform' or 'adaptive' or 'adversarial'
input_filename = None # give name to continue training from it
output_filename = 'base_%d_%d'%(train_problem_sizes[i][0], train_problem_sizes[i][1]) # 'base_6_6'
train_model(cl_problem_sizes, train_problem_sizes, maxTime, mode, input_filename, output_filename, classic=True)



# # ADAPTIVE training
# cl_problem_sizes = list(range(0,10)) # indices of problem sizes to choose from train_problem_sizes
# mode = 'adaptive' # 'uniform' or 'adaptive' or 'general'
# input_filename = None # give name to continue training from it
# output_filename = 'adaptive_cl'
# train_model(cl_problem_sizes, train_problem_sizes, maxTime, mode, input_filename, output_filename, adaptive_threshold)



# # ADVERSARIAL training
# cl_problem_sizes = list(range(0,10)) # indices of problem sizes to choose from train_problem_sizes
# mode = 'adversarial' # 'uniform' or 'adaptive' or 'adversarial'
# input_filename = None # give name to continue training from it
# output_filename = 'adversarial'
# adaptive_threshold = 0.15
# train_model(cl_problem_sizes, train_problem_sizes, maxTime, mode, input_filename, output_filename, adaptive_threshold)





