from Models.StateLSTM_test import StateLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from copy import deepcopy

class Rollout:
    def __init__(self,venv,actor,critic,device,masking):
        
        self.venv = venv
        self.actor = actor
        self.critic = critic
        self.device = device
        self.masking = masking
        
    def play(self,BS,BSind,training=True,size_beam_search=-1):
        
        total_reward = 0.0
        States = []
        Log_Prob = []
        Prob = []
        Action = []
        Value = []
        tr_reward = []
        entropies = []
        
        ite = 0
         
        while not self.venv.multidone:
        
            mac_utl = np.array([i['machine_utilization'] for i in self.venv.BState])
            job_time = np.array([i['job_times'] for i in self.venv.BState])                 # Duration matrix
            job_early = np.array([i['job_early_start_time'] for i in self.venv.BState])
            job_state = np.array(self.venv.BSjobstate)
            machine_state = np.array(self.venv.BSmachinestate)
            makespan = np.array(self.venv.BSmaxSpan)
            pre = np.array([i['precedence'] for i in self.venv.BState])                     # Machine matrix

            State = {
                'machine_utilization': mac_utl,
                'job_times': job_time,
                'job_early_start_time': job_early,
                'precedence': pre,
                'job_state': job_state,
                'machine_state': machine_state,
                'makespan': makespan
            }

            States.append(State)
 
            # Compute action
            with torch.set_grad_enabled(training):
                if ite==0 or size_beam_search > 1:
                    actorJobEmb = self.actor.instance_embed(State)
                    criticJobEmb = self.critic.instance_embed(State)

                prob, log_prob = self.actor(State,actorJobEmb,self.masking)
                value = self.critic(State,criticJobEmb)

            m = Categorical(prob)

            if training:
                action = m.sample().unsqueeze(1).cpu().numpy().tolist()
            else:
                action = torch.argmax(m.probs, dim=1).unsqueeze(1).cpu().numpy().tolist()
                
            entropy = m.entropy().cpu().detach().numpy()
            
            ID = [[i] for i in range(BS)]
            
            # Beam Search
            if size_beam_search >= 1:
                
                BSind = [i for i in range(size_beam_search)]
                argmax_prob = np.dstack(np.unravel_index(np.argsort(prob.ravel().cpu()), prob.shape))[0][::-1]
                ID = []
                action = []
                envs = []

                for k in range(size_beam_search):
                    ID.append([argmax_prob[k, 0]])
                    action.append([argmax_prob[k, 1]])
                    new_env = deepcopy(self.venv.envs[argmax_prob[k, 0]])
                    envs.append(new_env)

                self.venv.envs = envs
                
            log_prob = log_prob[ID,action]
            prob = prob[ID,action]

            # List append
            Action.append([i[0] for i in action])
            Prob.append(prob)
            Log_Prob.append(log_prob.squeeze(1))
            Value.append(value.squeeze(1))
            entropies.append(entropy)

            # Environment step
            self.venv.faststep(BSind,[i[0] for i in action])

            # Collect reward
            tr_reward.append(self.venv.BSreward)
            total_reward+=np.array(self.venv.BSreward)

            ite+=1
        
        
        return ite,total_reward,States,Log_Prob,Prob,Action,Value,tr_reward, np.mean(entropies)


class Actor(StateLSTM):
    def __init__(self, _embeddedDim, _jobs, _ops, _macs, device='cuda:0'):
        super(Actor, self).__init__(_embeddedDim, _jobs, _ops, _macs, device)
                
        # Actor network (not suitable for more than 1 block)
        self.Actor = nn.ModuleList([nn.ModuleDict({
            'a1': nn.Linear(9*self.embeddedDim,self.embeddedDim),
            'a2': nn.Linear(self.embeddedDim,16),
            'a3': nn.Linear(16,1)
        }) for i in range(1)])
        
    def instance_embed(self, State):
        JobEmbeddings = self.instanceEmbedding(State)
        
        return JobEmbeddings
      
    def forward(self, State, JobEmbeddings, masking):
        # input: State - OrderedDict
        #        masking - 0 or 1
        logsoft = nn.LogSoftmax(1)
        
        activation = F.leaky_relu
        
        stateEmbeded = self.dynamicEmbedding(State,JobEmbeddings)
                
        for l in self.Actor:
            LearnActor = l['a3'](activation(l['a2'](activation(l['a1'](stateEmbeded))))).squeeze(2)
            
        # option: do masking 
        # https://arxiv.org/abs/2006.14171 this paper directly set masked action to be a large negative number, say -1e+8
        if masking==1:
            invalid_action = (State['job_state']==self.ops)*1.0  # BS, num_jobs  
            LearnActor -= torch.tensor(invalid_action*1e+30,dtype=torch.float64,device=self.device)
            
        prob = F.softmax(LearnActor,dim=1) # BS, num_jobs
        log_prob = logsoft(LearnActor)    
        return prob, log_prob


class Critic6(StateLSTM):
    def __init__(self, _embeddedDim, _jobs, _ops, _macs, device='cuda:0'):
        super(Critic6, self).__init__(_embeddedDim, _jobs, _ops, _macs, device)
                
        # Critic network (not suitable for more than 1 block)
        self.Critic = nn.ModuleList([nn.ModuleDict({
            'proj': nn.Linear(9*self.embeddedDim,self.embeddedDim),
            'c1': nn.Linear(self.embeddedDim,16),
            'c2': nn.Linear(16,1)
        }) for i in range(1)])
                                 
        self.attn = nn.Linear(2*self.embeddedDim,self.embeddedDim)
        self.attn_proj = nn.Linear(self.embeddedDim,1)
        
    def instance_embed(self, State):
        JobEmbeddings = self.instanceEmbedding(State)
        
        return JobEmbeddings
      
    def forward(self, State, JobEmbeddings):
        # input: State - OrderedDict
        
        activation = F.leaky_relu
        
        stateEmbeded = self.dynamicEmbedding(State,JobEmbeddings)
        
        for l in self.Critic:
            
            valueRaw = l['proj'](stateEmbeded) # reduce size - no activation - BS, jobs, embeddim
            
            valueMean = torch.mean(valueRaw,dim=1,keepdim=True).repeat(1,self.jobs,1) # BS, jobs, embeddim
            
            value = torch.cat([valueRaw,valueMean],dim=2) # BS, jobs, 2*embeddim            
            Attn = activation(self.attn(value)) # BS, jobs, embeddim
            Attn = F.softmax(self.attn_proj(Attn),dim=1) # BS, jobs, 1
            value = torch.sum(valueRaw*Attn,dim=1) # BS, embeddim
            value = l['c2'](activation(l['c1'](activation(value)))) # BS, 1

        return value  
