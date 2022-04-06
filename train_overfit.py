import time
import numpy as np
import torch
from torch import optim
from utils import *
from Envs.JobShopMultiGymEnv import *
from Models.actorcritic import *

embeddim = 128
actorLR = 1e-4
criticLR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

jobs = 15
ops = 15
macs = ops
maxTime = 100
i = 0
size_agnostic = True
resume_run = True
seed = 0
BS = 128

start = time.time()
########################################################################

print('ta{}'.format(i + 1))
test_precedence, test_timepre_ = read_instances('./data/ta/ta{}'.format(i + 1))
test_timepre = test_timepre_ / maxTime

# Create batch
test_precedence = np.repeat(test_precedence, BS, axis=0)
test_timepre = np.repeat(test_timepre, BS, axis=0)

test_venv = JobShopMultiGymEnv(BS, jobs, ops, macs)
test_venv.setGame(test_precedence, test_timepre)
testSamples = [i for i in range(BS)]

filedir = './Results/%dx%d/' % (jobs, macs)
if size_agnostic:
    filedir = './Results/size_agnostic/'

model_name = 'A2C_Seed%d_BS%d.tar' % (seed, BS)
if size_agnostic:
    model_name = '6_6_100.tar'

# Instanitate actor-critic
actor = Actor(embeddim, jobs, ops, macs, device).to(device)
critic = Critic(embeddim, jobs, ops, macs, device).to(device)

# Environment training

actor_opt = optim.Adam(actor.parameters(), lr=actorLR)
critic_opt = optim.Adam(critic.parameters(), lr=criticLR)

if resume_run:
    checkpoint = torch.load(filedir + model_name, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
    critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

for epi in range(10000):

    if epi%10==1 or epi==0:
        epi_st = time.time()

    test_venv.reset(testSamples)

    train_rollout = Rollout(test_venv, actor, critic, device)
    ite, total_reward, States, Log_Prob, Prob, Action, Value, tr_reward, tr_entropy = train_rollout.play(BS,testSamples)
    Log_Prob = torch.stack(Log_Prob)

    tr_reward.reverse()

    #Compute Advantage: Qvalue - Value
    Qvalue = np.zeros((ite, BS))
    q = 0.0
    for i in range(ite):
        q += np.array(tr_reward[i])  # q.shape=[BS]
        Qvalue[ite - i - 1] = q

    Advantage = torch.tensor(Qvalue, device=device) - torch.stack(Value)  # advantage.shape=[ite,BS]

    # Zero grad
    actor.zero_grad()
    critic.zero_grad()

    # Compute loss fucntion
    # actor_loss = torch.mean(torch.sum(Log_Prob, dim=1)*torch.tensor(total_reward).to(device))
    actor_loss = torch.mean(-Log_Prob * Advantage.detach())
    critic_loss = 0.5 * torch.mean(Advantage ** 2)

    # Update weights Actor and critic
    actor_loss.backward()
    critic_loss.backward()
    actor_opt.step()
    critic_opt.step()

    if epi % 10 == 0:

        epi_et = time.time()
        
        print('epi: %d, al: %.2e, cl: %.2e, trSpan: %.4f, trEntropy: %.2f, time: %.2f' \
              % (epi, actor_loss.data.item(), critic_loss.data.item(), np.mean(total_reward), tr_entropy, epi_et-epi_st))

