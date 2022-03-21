# parent class for state representation

import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Models.Set2Set_test import Set2Set

class StateLSTM(nn.Module):
    def __init__(self, _embeddedDim, _jobs, _ops, _macs, device='cuda:0'):
        super(StateLSTM, self).__init__()
        self.embeddedDim = _embeddedDim
        self.jobs = _jobs
        self.ops = _ops
        self.macs = _macs
        self.device = device

        # static embedding
        self.machinesEmbedding = nn.Linear(1, self.embeddedDim)
        self.jobTimeEmbedding = nn.Linear(1, self.embeddedDim)
        
        self.sequenceLSTM = nn.LSTM(2*self.embeddedDim, self.embeddedDim, batch_first=True) # set batch size as 1st Dim
    
        # dynamic representation layers
        self.jobStartTimeEmbed = nn.Linear(1, self.embeddedDim)
        self.machineTimeEmbed = nn.Linear(1, self.embeddedDim)
        self.dynEmbedding = nn.Linear(7, 2*self.embeddedDim)
        
        # activation function
        self.activation = F.leaky_relu

        # Set2Set function to get inter Job Embeddings
        self.interJobEmbedding = Set2Set(3*self.embeddedDim, 1, 1)
        
    def numel(self):
        
        return np.sum([torch.numel(w) for w in self.parameters()])
       
    def instanceEmbedding(self,State):
        
        Job_times = torch.tensor(State['job_times'],device=self.device,dtype=torch.float64)
        Precedences = torch.tensor(State['precedence'],device=self.device,dtype=torch.int64)
        BS = Precedences.shape[0]
            
        # augment input to add individual job terminal state
        Precedences_extra = self.macs*torch.ones(BS,self.jobs,1,device=self.device,dtype=torch.int64)
        Job_times_extra = torch.zeros(BS,self.jobs,1,device=self.device,dtype=torch.float64)
               
        # augmented input
        Precedences = torch.cat((Precedences,Precedences_extra),dim=2) # BS, num_jobs, num_ops+1
        Job_times = torch.cat((Job_times,Job_times_extra),dim=2) # BS, num_jobs, num_ops+1
        
        # embedding                      
        Job_times = self.jobTimeEmbedding(Job_times.unsqueeze(3))
        Precedences_float = Precedences.unsqueeze(3).to(dtype=torch.float64)
        Precedences = self.machinesEmbedding(Precedences_float) # Zangir size agnostic

        # concat embeded precedence and job time - input has to be a 3D tensor: batch, seq, feature
        PrecedenceTime =torch.cat((Precedences,Job_times),dim=3).reshape(BS*self.jobs,self.ops+1,-1) 
        
        # LSTM embedding 
        JobEmbeddings,_ = self.sequenceLSTM(torch.flip(PrecedenceTime,[1])) # did a flip for reverse sequence
        JobEmbeddings = JobEmbeddings.reshape(BS,self.jobs,self.ops+1,-1) # shape - BS, num_jobs, num_ops+1, embed_dim

        return JobEmbeddings
    
    def dynamicEmbedding(self,State,JobEmbeddings):
        # 
        # dynamic representation
        #
        
        #Job_early_start_time = torch.tensor(State['job_early_start_time'],dtype=torch.float64,device=self.device).unsqueeze(2)
        BS = JobEmbeddings.shape[0]
        
        Machine_utilization = torch.tensor(State['machine_utilization'],dtype=torch.float64,device=self.device).unsqueeze(2)
        # add extra machine for when job is finished
        Machine_utilization_extra = torch.zeros(BS,1,1,dtype=torch.float64,device=self.device)
        Machine_utilization = torch.cat((Machine_utilization,Machine_utilization_extra),dim=1) #BS, mac+1, 1 
        
          
        BSID = [[i] for i in range(BS)]
        JobID = [[i for i in range(self.jobs)] for j in range(BS)]
             
        JobEmbeddings = JobEmbeddings[BSID,JobID,self.ops-State['job_state'],:]
        
        #
        # Hand-crafted features for operations
        #
        precedence = torch.tensor(State['precedence'], device=self.device)
        job_times = torch.tensor(State['job_times'], device=self.device)
        job_state = torch.tensor(State['job_state'], device=self.device)

        job_times_extra = torch.zeros(BS,self.jobs,1,dtype=torch.float64,device=self.device)
        job_times = torch.cat((job_times,job_times_extra),dim=2) 


        job_processing_time = torch.gather(job_times, 2, job_state.unsqueeze(2))                                    # Processing time of next operation [BS, jobs, 1]
        job_start_time      = torch.tensor(State['job_early_start_time'], dtype=torch.float64,  device=self.device).unsqueeze(2)          # Time for the next operation to start [BS, jobs, 1]
        job_end_time        = job_start_time + job_processing_time                                                  # Time for the next operation to finish [BS, jobs, 1]
        total_work_remaining= torch.flip(torch.cumsum(torch.flip(job_times, dims=[2]), dim=2), dims=[2])            
        total_work_remaining= torch.gather(total_work_remaining, 2, job_state.unsqueeze(2))                         # Cumulative time remaining per job [BS, jobs, 1]
        number_pending_ops  = torch.tensor(self.ops-State['job_state'], device=self.device).unsqueeze(2)            # Number of pending operations per job [BS, jobs, 1]


        # editing precedence
        MacID = np.concatenate((State['precedence'], self.macs*np.ones([BS,self.jobs,1])), axis=2)
        MacID = MacID[BSID,JobID,State['job_state']]
        

        machine_state = torch.tensor(State['machine_state'],dtype=torch.float64,device=self.device).unsqueeze(2)
        # add extra machine for when job is finished
        machine_state_extra = torch.zeros(BS,1,1,dtype=torch.float64,device=self.device)
        machine_state = torch.cat((machine_state,machine_state_extra),dim=1) #BS, mac+1, 1 

        #
        # Hand-crafted features for machines
        #
        Machine_utilization = Machine_utilization[BSID,MacID,:]                                                    # Machine accumulate utilization (associated to each job) [BS, jobs, 1]
        ordered_times = torch.zeros_like(job_times).scatter_(dim=2, index=precedence, src=job_times)
        total_work_machine = torch.sum(ordered_times, dim=1).unsqueeze(2)
        total_remaining_machine = total_work_machine[BSID,MacID,:]
        total_remaining_machine = total_remaining_machine - Machine_utilization                                    # Work remaining per machine (associated to each job)  [BS, jobs, 1]
        number_pending_ops_machine = (self.ops - machine_state)[BSID,MacID,:]                                      # Number of pernding operations per machine (associated to each job) [BS, jobs, 1]
        current_makespan = torch.tensor(State['makespan'],dtype=torch.float64,\
                            device=self.device).unsqueeze(1).repeat(1,self.jobs).unsqueeze(2)                      # Current makespan per job [BS, jobs, 1]

        # embedding 
        dynamic_feats = torch.cat((job_processing_time,job_start_time,job_end_time,total_work_remaining, Machine_utilization,total_remaining_machine, current_makespan), dim=2)
        dynamic_featsEmb = self.dynEmbedding(dynamic_feats)

        # embedding 
        #job_start_time = self.jobStartTimeEmbed(job_start_time) # BS, num_jobs, emded_dim
        #Machine_utilization = self.machineTimeEmbed(Machine_utilization) # BS, num_mac+1, emded_dim
         
        #stateEmbeded = torch.cat((JobEmbeddings,Machine_utilization,job_start_time),dim=2)
        stateEmbeded = torch.cat((JobEmbeddings, dynamic_featsEmb),dim=2)

        # Set2set model between jobs
        stateEmbeded = self.interJobEmbedding(stateEmbeded)
                                   
        return stateEmbeded
