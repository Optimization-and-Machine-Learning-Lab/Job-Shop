from pickle import TRUE
import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import itertools
from collections import OrderedDict

from Envs.JobShopMultiGymEnv import *
from Models.StateLSTM import *
from Models.actorcritic import *
from copy import deepcopy
from create_instances import create_test

torch.cuda.empty_cache() # NEW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import contextlib
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# create instances for 6,6 and 10,10 and sove using
# https://developers.google.com/optimization/scheduling/job_shop

# get instances of taillard with solutions
# save ideal solutions means

optimal_test_rewards = [
                    530.0, # 6x6
                    1000.0, # 10x10
                    1228.9, # 15x15
                    1364.9, # 20x15
                    1617.3, # 20x20
                    1790.1, # 30x15
                    1948.3, # 30x20
                    2773.8, # 50x15
                    2843.9, # 50x20
                    5365.7 # 100x20
                    ]

train_problem_sizes = [[6,6], [10,10], [15,15], [20,15], [20,20], [30,15], [30,20], [50,15], [50,20], [100,20]] 
Batch = [256, 128, 128, 128, 128, 64, 64, 32, 32, 8]

def train_model(problem_sizes, train_problem_sizes, maxTime, mode, input_filename, output_filename, adaptive_threshold=0.0, classic=False):

    embeddim = 128
    actorLR = 1e-4
    criticLR = 1e-4
    Episode = 45000 # total number of episodes
    testsize = 10
    best_model_reward = -999999
    seed = 0
    masking = 1
    cl_test_data = []
    text_file = './output_logs/txt/' + output_filename + '.txt'
    csv_file = './output_logs/csv/train/' + output_filename + '.csv'
    output_model = './output_models/' + output_filename + '.tar'
    adaptive_current_problem = 0

    if input_filename != None:
        input_filename = './output_models/' + input_filename + '.tar'

    with open(text_file, 'a') as f:
        print('device: ', device, file=f)
        print('problem_sizes: ', problem_sizes, file=f)

    problem_sizes_test = range(len(train_problem_sizes))
    for k in problem_sizes_test:

        jobs = train_problem_sizes[k][0]
        macs = train_problem_sizes[k][1]
        ops = macs

        testdata = torch.load('./data/JS_test_%d_%dx%d_t%d.tar'%(testsize,jobs,macs,maxTime))
        test_precedence = testdata['precedence']
        test_timepre = testdata['time_pre']/float(maxTime)

        test_venv = JobShopMultiGymEnv(testsize,jobs,ops,macs)
        test_venv.setGame(test_precedence,test_timepre)
        testSamples = [i for i in range(testsize)]

        cl_test_data.append([deepcopy(test_venv), testSamples.copy()])
      
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Instanitate actor-critic
    actor = Actor(embeddim,jobs,ops,macs,device).to(device)
    critic = Critic(embeddim,jobs,ops,macs,device).to(device)
    
    actor_opt = optim.Adam(actor.parameters(), lr=actorLR)
    critic_opt = optim.Adam(critic.parameters(), lr=criticLR)

    if input_filename!=None:
        try:
            checkpoint = torch.load(input_filename, map_location=device)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
            actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
            critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        except:
            with open(text_file, 'a') as f:
                print("Something went wrong when opening the file.", file=f)

    # Start training from the begining                        

    TeReward = []
    tr_numstep = 0
    
    with open(text_file, 'a') as f:
        print ('Seed: %d, Invalid-action-masking: %d' %(seed,masking), file=f)

    epi_start = 0
    epi_end = Episode+1
    step_ite = 0
    same_adaptive_size_iter = 0
    general_probs = [0] * len(problem_sizes_test)
    
    for epi in range(epi_start,epi_end):

        Advantage = []
        
        if epi%100==1 or epi==0:
            epi_st = time.time()

        if mode == 'adaptive':
            base_case = np.random.randint(2)
            if base_case:
                problem_size_ind = adaptive_current_problem
            else:
                problem_size_ind = np.random.randint(0, adaptive_current_problem + 1)
            problem_size_ind = int(problem_size_ind)
        elif mode == 'adversarial':
            current_probs = general_probs[:adaptive_current_problem + 1]

            distrib_probs = softmax(current_probs)
            problem_size_ind = np.random.choice(adaptive_current_problem + 1, size=1, p=distrib_probs)
            problem_size_ind = int(problem_size_ind)
        else:
            problem_size_ind = np.random.randint(0, len(problem_sizes))

        jobs = train_problem_sizes[problem_sizes[problem_size_ind]][0]
        macs = train_problem_sizes[problem_sizes[problem_size_ind]][1]
        ops = macs

        BS = Batch[problem_sizes[problem_size_ind]]
        actor.jobs, critic.jobs, = jobs, jobs
        actor.ops, critic.ops, = ops, ops
        actor.macs, critic.macs, = macs, macs

        with temp_seed((testsize+epi+1)*(seed+1)):
            precedence = ([[np.random.choice(ops,ops,replace=False).tolist() for i in range(jobs)] for j in range(BS)])
            time_pre = np.random.randint(1,maxTime,size=(BS,jobs,ops))/float(maxTime)

        train_venv = JobShopMultiGymEnv(BS,jobs,ops,macs)

        # Create training virtual environment    
        train_venv.setGame(precedence,time_pre)
        trainSample = [i for i in range(BS)]
        train_venv.reset(trainSample)   

        step_ite += 1

        # Rollout
        train_rollout = Rollout(train_venv,actor,critic,device,masking)
        ite,total_reward,States,Log_Prob,Prob,Action,Value,tr_reward, tr_entropy = train_rollout.play(BS,trainSample)
        tr_numstep+=ite

        Log_Prob = torch.stack(Log_Prob)

        # !!!!!!!!!!Compute Advantage: Qvalue - Value
        tr_reward.reverse()

        Qvalue = np.zeros((ite,BS))
        q = 0.0
        for i in range(ite):
            q+=np.array(tr_reward[i])           # q.shape=[BS] 
            Qvalue[ite-i-1] = q

        Advantage = torch.tensor(Qvalue,device=device) - torch.stack(Value)         #advantage.shape=[ite,BS]

        # Zero grad
        actor.zero_grad()
        critic.zero_grad()

        # Compute loss fucntion
        actor_loss = torch.mean(-Log_Prob*Advantage.detach())
        critic_loss = 0.5*torch.mean(Advantage**2)

        # Update weights Actor and critic
        actor_loss.backward()
        critic_loss.backward()
        actor_opt.step()
        critic_opt.step()

        if epi%100 == 0:    

            epi_et = time.time() 

            rewards = []

            updated = False

            for k in range(len(problem_sizes_test)):    

                jobs = train_problem_sizes[problem_sizes_test[k]][0]
                macs = train_problem_sizes[problem_sizes_test[k]][1]
                ops = macs

                actor.jobs, critic.jobs, = jobs, jobs
                actor.ops, critic.ops, = ops, ops
                actor.macs, critic.macs, = macs, macs

                test_venv, testSamples = cl_test_data[k]
                test_venv.reset(testSamples)
                test_rollout = Rollout(test_venv,actor,critic,device,masking)

                te_ite,te_total_reward,te_States,te_Log_Prob,te_Prob,te_Action,te_Value,te_reward, te_entropy\
                = test_rollout.play(testsize,testSamples,False)

                test_reward = np.mean(te_total_reward)
                thresh_reward = optimal_test_rewards[problem_sizes_test[adaptive_current_problem]] * (1+adaptive_threshold)
                reward_to_comare = abs(test_reward * 100)
                general_probs[k] = (reward_to_comare - thresh_reward)/thresh_reward

                if mode == 'adaptive' or mode == 'adversarial':
                    same_adaptive_size_iter += 100
                    if k == adaptive_current_problem and not updated:
                        if reward_to_comare <= thresh_reward or (mode == 'adversarial' and same_adaptive_size_iter >= 5000): 
                            adaptive_current_problem += 1
                            lastModel = {
                                        'actor_state_dict': actor.state_dict(),
                                        'critic_state_dict': critic.state_dict(),
                                        'actor_opt_state_dict': actor_opt.state_dict(),
                                        'critic_opt_state_dict': critic_opt.state_dict(),
                                        }
                            torch.save(lastModel,output_model)
                            same_adaptive_size_iter = 0

                        elif mode == 'adversarial' and same_adaptive_size_iter>=3000: # K

                            adaptive_current_problem = adaptive_current_problem + 1 # k up
                            adaptive_current_problem = int(adaptive_current_problem)
                            same_adaptive_size_iter = 0


                        updated = True

                TeReward.append(test_reward)
                rewards.append(test_reward)

                with open(text_file, 'a') as f:
                    print('epi: %d, al: %.2e, cl: %.2e, trEntropy: %.2f, teSpan: %.4f, tr_step: %d, time: %.2f, jobs : %d, ops : %d, adaptive_current_problem : %d'\
                    %(epi,actor_loss.data.item(),critic_loss.data.item(), tr_entropy, test_reward,tr_numstep,epi_et-epi_st,jobs,ops,adaptive_current_problem), file=f)

                with open(csv_file, 'a') as f:
                    print('%d, %.2e,%.2e, %.4f'\
                    %(epi,actor_loss.data.item(),critic_loss.data.item(), test_reward), file=f)
                
            # Save model
            if not classic:
                test_reward = np.mean(rewards)
                if test_reward > best_model_reward and mode == 'uniform':
                    best_model_reward = test_reward
                    lastModel = {
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'actor_opt_state_dict': actor_opt.state_dict(),
                    'critic_opt_state_dict': critic_opt.state_dict(),
                    }
                    torch.save(lastModel,output_model)
            else:
                if rewards[problem_sizes[0]]:
                    best_model_reward = test_reward
                    lastModel = {
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'actor_opt_state_dict': actor_opt.state_dict(),
                    'critic_opt_state_dict': critic_opt.state_dict(),
                    }
                    torch.save(lastModel,output_model)

            if adaptive_current_problem == 10:
                break



