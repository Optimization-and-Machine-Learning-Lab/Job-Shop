import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from copy import deepcopy
from Models.StateLSTM import StateLSTM

class Rollout:
    def __init__(self,venv,actor,critic,device):
        
        self.venv = venv
        self.actor = actor
        self.critic = critic
        self.device = device
        
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
        done = False
        while not done:
        
            state = self.venv.bstate
            States.append(state)
 
            # Compute action
            with torch.set_grad_enabled(training):
                if ite==0 or size_beam_search > 1:
                    actorJobEmb = self.actor.instance_embed(state)
                    criticJobEmb = self.critic.instance_embed(state)

                prob, log_prob = self.actor(state,actorJobEmb)
                value = self.critic(state,criticJobEmb)

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
            state, reward, done = self.venv.step(BSind,[i[0] for i in action])

            # Collect reward
            tr_reward.append(reward)
            total_reward+=np.array(reward)

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
      
    def forward(self, State, JobEmbeddings):

        logsoft = nn.LogSoftmax(1)
        
        activation = F.leaky_relu
        
        stateEmbeded = self.dynamicEmbedding(State,JobEmbeddings)
                
        for l in self.Actor:
            LearnActor = l['a3'](activation(l['a2'](activation(l['a1'](stateEmbeded))))).squeeze(2)
            
        # Masking
        invalid_action = (State['job_state']==self.ops)*1.0  # BS, num_jobs  
        LearnActor -= torch.tensor(invalid_action*1e+30,dtype=torch.float64,device=self.device)
            
        prob = F.softmax(LearnActor,dim=1) # BS, num_jobs
        log_prob = logsoft(LearnActor)    
        return prob, log_prob


class Critic(StateLSTM):
    def __init__(self, _embeddedDim, _jobs, _ops, _macs, device='cuda:0'):
        super(Critic, self).__init__(_embeddedDim, _jobs, _ops, _macs, device)
                
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
