import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import itertools
from collections import OrderedDict

from Envs.JobShopMultiGymEnv import *
from Models.StateLSTM_test import *
from Models.actorcritic import *
from config import read_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Arguments 
args = read_args()
testsize = 1000
jobs = args['jobs']
macs = args['macs']
ops = macs
maxTime = args['maxTime']


# Load testing dataset:
testdata = torch.load('./data/JS_test_%d_%dx%d_t%d.tar'%(testsize,jobs,macs,maxTime))
test_precedence = testdata['precedence']
test_timepre = testdata['time_pre']/float(maxTime)
testSample = [i for i in range(testsize)]



import contextlib
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


# Hyperparameters
embeddim = 128
actorLR = 1e-4
criticLR = 1e-4
masking = 1
Episode = 45000 # total number of episodes

# Training 
resume_run=False
Seed = [0]
Batch = [100]
Mask = [1]

filedir = './Results/%dx%d/'%(jobs,macs)


# Learning loop...
for seed,BS,masking in itertools.product(Seed,Batch,Mask):
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model_name = 'A2C_Seed%d_BS%d_Mask%d.tar'%(seed,BS,masking)
    
    JS_best = OrderedDict()
    JS_result = OrderedDict()
    
    # Instanitate actor-critic
    actor = Actor(embeddim,jobs,ops,macs,device).to(device)
    critic = Critic6(embeddim,jobs,ops,macs,device).to(device)
    
    # Environment test
    test_venv = JobShopMultiGymEnv(testsize,jobs,ops,macs)
    test_venv.setGame(test_precedence,test_timepre)
    testSamples = [i for i in range(testsize)]
    test_venv.reset(testSamples)


    
    # Environment training
    train_venv = JobShopMultiGymEnv(BS,jobs,ops,macs) #put on top - reduce time
    
    actor_opt = optim.Adam(actor.parameters(), lr=actorLR)
    critic_opt = optim.Adam(critic.parameters(), lr=criticLR)

    if resume_run:
        try:
            checkpoint = torch.load(filedir+model_name, map_location=device)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
            actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
            critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        except:
            print("Something went wrong when opening the file.")

    # Instantiate rollouts
    test_rollout = Rollout(test_venv,actor,critic,device,masking)
    

    # Start training from the begining                        
    
    StoppedEpi = 0
    valReward = []
    TeReward = []
    tr_numstep = 0

    steps_list = []
    
    print ('Seed: %d, Batch-Size: %d, Invalid-action-masking: %d' %(seed,BS,masking))

    epi_start = 0
    epi_end = Episode+1
    start_step = ops
    step_ite = 0
    
    for epi in range(epi_start,epi_end):
        
        States = []
        Advantage = []
        total_reward = np.zeros(BS)
        
        if epi%100==1 or epi==0:
            epi_st = time.time()

        # Generate random mini-batch
        with temp_seed((testsize+epi+1)*(seed+1)):
            precedence = ([[np.random.choice(ops,ops,replace=False).tolist() for i in range(jobs)] for j in range(BS)])
            time_pre = np.random.randint(1,maxTime,size=(BS,jobs,ops))/float(maxTime)

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
        #actor_loss = torch.mean(torch.sum(Log_Prob, dim=1)*torch.tensor(total_reward).to(device))
        actor_loss = torch.mean(-Log_Prob*Advantage.detach())
        critic_loss = 0.5*torch.mean(Advantage**2)

        # Update weights Actor and critic
        actor_loss.backward()
        critic_loss.backward()
        actor_opt.step()
        critic_opt.step()

        
        # 
        if epi%100 == 0:         

            #Test rollout
            test_venv.reset(testSamples)

            te_ite,te_total_reward,te_States,te_Log_Prob,te_Prob,te_Action,te_Value,te_reward, te_entropy\
            = test_rollout.play(testsize,testSamples,False)

            TeReward.append(np.mean(te_total_reward))
            
            # ...toc
            epi_et = time.time()
            
            print('epi: %d, al: %.2e, cl: %.2e, trSpan: %.4f, trEntropy: %.2f, teSpan: %.4f, tr_step: %d, time: %.2f'\
                   %(epi,actor_loss.data.item(),critic_loss.data.item(), np.mean(total_reward), tr_entropy, np.mean(te_total_reward),tr_numstep,epi_et-epi_st))
            
            # Save model
            lastModel = {
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_opt_state_dict': actor_opt.state_dict(),
            'critic_opt_state_dict': critic_opt.state_dict(),
            }
            torch.save(lastModel,filedir+model_name)



