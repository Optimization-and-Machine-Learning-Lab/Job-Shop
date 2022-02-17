
from Envs.JobShopMultiGymEnv import *
#from main import *
from utils import *
from Models.actorcritic import *
from torch import optim

embeddim = 128
#actorLR = 1e-4
#criticLR = 1e-4
masking = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


testsize = 1
jobs = 15
ops = 15
macs = 15
maxTime = 100

seed = 0
BS = 100
masking = 1


########################################################################

test_precedence,test_timepre_ =read_instances('./data/ta/ta1')
test_timepre = test_timepre_ / maxTime


test_venv = JobShopMultiGymEnv(testsize,jobs,ops,macs)
test_venv.setGame(test_precedence,test_timepre)
testSamples = [i for i in range(testsize)]
test_venv.reset(testSamples)

# Instanitate actor-critic
actor = Actor(embeddim,jobs,ops,macs,device).to(device)
critic = Critic6(embeddim,jobs,ops,macs,device).to(device)

# Environment training

actor_opt = optim.Adam(actor.parameters())
critic_opt = optim.Adam(critic.parameters())

filedir = './Results/%dx%d/'%(jobs,macs)
model_name = 'A2C_Seed%d_BS%d_Mask%d.tar'%(seed,BS,masking)

checkpoint = torch.load(filedir+model_name, map_location=torch.device('cpu'))
actor.load_state_dict(checkpoint['actor_state_dict'])
critic.load_state_dict(checkpoint['critic_state_dict'])
actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])



test_rollout = Rollout(test_venv,actor,critic,device,masking)
te_ite,te_total_reward,te_States,te_Log_Prob,te_Prob,te_Action,te_Value,te_reward, te_entropy = test_rollout.play(testsize,testSamples,False)

print(te_total_reward)

