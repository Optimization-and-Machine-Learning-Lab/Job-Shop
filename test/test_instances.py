
from Envs.JobShopMultiGymEnv import *
from utils import *
from Models.actorcritic import *
from torch import optim
import time

embeddim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change
instances = 'ta'            # Select: ta or dmu
size_agnostic = True
testsize = 1
jobs = 15
macs = 15
ops = macs 
maxTime = 100               # maxTime: ta -> 100, for dmu -> 200
seed = 0
BS = 128
masking = 1
beam_size = 2
model_jobs = 6
model_ops = 6

start = time.time()
########################################################################

for i in range(0,10):

    
    print('{}{}'.format(instances,i+1))

    if instances=='ta':
        test_precedence,test_timepre_ = read_instances('./data/ta/ta{}'.format(i+1))
    else:
        test_precedence,test_timepre_ = read_instances('./data/dmu/dmu{}.txt'.format(i+1))

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

    if size_agnostic:
        filedir = './Results/size_agnostic/%dx%d_%d/'%(model_jobs,model_ops,maxTime)

    model_name = 'A2C_Seed%d_BS%d_Mask%d.tar'%(seed,BS,masking)

    checkpoint = torch.load(filedir+model_name, map_location=torch.device('cpu'))
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
    critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

    test_rollout = Rollout(test_venv,actor,critic,device,masking)
    te_ite,te_total_reward,te_States,te_Log_Prob,te_Prob,te_Action,te_Value,te_reward, te_entropy = test_rollout.play(testsize,testSamples,False,beam_size)

    print(te_total_reward)

end = time.time()
#print(end - start)

