
from Envs.JobShopMultiGymEnv import *
from utils import *
from Models.actorcritic import *
from Models.RolloutBeam import *
from Models.RolloutPOMO import *
from Models.RolloutSampling import *
from torch import optim

import time

def test_model(problem_sizes, maxTime, data_mode, search_mode, input_model, output_log):

    embeddim = 128
    masking = 1
    testsize = 1
    output_csv = './output_logs/csv/test/' + output_log + '.csv'
    filedir = './output_models/'
    input_model = filedir + input_model + '.tar'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_mode == 'ta':
        max_data_size = 80

    for i in range(max_data_size):

        if data_mode == 'ta':
            problem_ind = i//10 + 2
            data = read_instances('./data/ta/ta{}'.format(i+1))
            jobs = problem_sizes[problem_ind][0]
            macs = problem_sizes[problem_ind][1]
        ops = macs

        test_precedence,test_timepre_ = data
        test_timepre = test_timepre_ / maxTime

        test_venv = JobShopMultiGymEnv(testsize,jobs,ops,macs)
        test_venv.setGame(test_precedence,test_timepre)
        testSamples = [i for i in range(testsize)]
        test_venv.reset(testSamples)

        # Instanitate actor-critic
        actor = Actor(embeddim,jobs,ops,macs,device).to(device)
        critic = Critic(embeddim,jobs,ops,macs,device).to(device)

        # Environment training

        actor_opt = optim.Adam(actor.parameters())
        critic_opt = optim.Adam(critic.parameters())

        checkpoint = torch.load(input_model, map_location=torch.device('cpu'))
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

        if search_mode == 'beam':
            test_rollout = RolloutBeam(test_venv,actor,critic,device,masking)
            size_search = 2
        elif search_mode == 'pomo':
            test_rollout = RolloutPOMO(test_venv,actor,critic,device,masking)
            size_search = 128
        elif search_mode == 'sampling':
            test_rollout = RolloutSampling(test_venv,actor,critic,device,masking)
            size_search = 128
        elif search_mode == 'greedy':
            test_rollout = RolloutBeam(test_venv,actor,critic,device,masking)
            size_search = 1

        te_ite,te_total_reward,te_States,te_Log_Prob,te_Prob,te_Action,te_Value,te_reward, te_entropy = test_rollout.play(testsize,testSamples,False,size_search)
        test_reward = np.max(te_total_reward)

        with open(output_csv, 'a') as f:
            print('%d, %.4f'%(te_ite,test_reward), file=f)

