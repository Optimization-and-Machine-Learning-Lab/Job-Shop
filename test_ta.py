from Envs.JobShopMultiGymEnv import *
from utils import *
from Models.actorcritic import *
from torch import optim

import time
import numpy as np
import torch

embeddim = 128
# actorLR = 1e-4
# criticLR = 1e-4
masking = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testsize = 1
jobs = 15
ops = 15
macs = ops
maxTime = 100
i = 0
size_agnostic = True

seed = 0
BS = testsize
masking = 1

start = time.time()
########################################################################

print('ta{}'.format(i + 1))
test_precedence, test_timepre_ = read_instances('./data/ta/ta{}'.format(i + 1))

test_timepre = test_timepre_ / maxTime

test_venv = JobShopMultiGymEnv(testsize, jobs, ops, macs)
test_venv.setGame(test_precedence, test_timepre)
testSamples = [i for i in range(testsize)]

if size_agnostic:
    filedir = './Results/size_agnostic/'

model_name = 'A2C_Seed%d_BS%d_Mask%d.tar' % (seed, BS, masking)

if size_agnostic:
    model_name = '%d_%d_100.tar' % (jobs, macs)

checkpoint = torch.load(filedir + model_name, map_location=device)

# Instanitate actor-critic
actor = Actor(embeddim, jobs, ops, macs, device).to(device)
critic = Critic6(embeddim, jobs, ops, macs, device).to(device)

# Environment training

actor_opt = optim.Adam(actor.parameters())
critic_opt = optim.Adam(critic.parameters())

filedir = './Results/%dx%d/' % (jobs, macs)
model_name = 'A2C_Seed%d_BS%d_Mask%d.tar' % (seed, BS, masking)

actor.load_state_dict(checkpoint['actor_state_dict'])
critic.load_state_dict(checkpoint['critic_state_dict'])
actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

# actor_opt.param_groups[0]['lr'] = 1e-6
# critic_opt.param_groups[0]['lr'] = 1e-6

for epi in range(10000):

    test_venv.reset(testSamples)

    train_rollout = Rollout(test_venv, actor, critic, device, masking)
    ite, total_reward, States, Log_Prob, Prob, Action, Value, tr_reward, tr_entropy = train_rollout.play(testsize,
                                                                                                         testSamples)
    Log_Prob = torch.stack(Log_Prob)

    # !!!!!!!!!!Compute Advantage: Qvalue - Value
    tr_reward.reverse()

    Qvalue = np.zeros((ite, testsize))
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

    #
    if epi % 10 == 0:
        print('epi: %d, al: %.2e, cl: %.2e, trSpan: %.4f, trEntropy: %.2f' \
              % (epi, actor_loss.data.item(), critic_loss.data.item(), np.mean(total_reward), tr_entropy))

