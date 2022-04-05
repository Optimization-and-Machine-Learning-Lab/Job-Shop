import gym
import numpy as np

class JobShopGymEnv(gym.Env):
    
    def __init__(self, maxJobs = 10, maxJobLength=10, n_machines = 10):
        super(JobShopGymEnv, self).__init__()

        self.maxJobs = maxJobs
        self.maxJobLength = maxJobLength
        self.n_machines = n_machines

    def setGame(self,precedence,time_pre):
        self.precedenceInp=precedence
        self.time_preInp=np.array(time_pre,dtype=np.float32)
        
    def reset(self):

        self.state = {}
        self.state['job_times']=np.array(self.time_preInp)
        self.state['machine_utilization'] = np.zeros(self.n_machines)
        self.state['job_early_start_time'] = np.zeros(self.maxJobs)
        self.state['precedence']=np.array(self.precedenceInp)

        self.jobsState = np.zeros(self.maxJobs,dtype=int)
        self.maxSpan = 0.0
        self.reward = 0.0
        self.jobsToProcess = self.maxJobs
        self.jobDone=np.zeros(self.maxJobs)
        self.done=False

        self.placement = np.zeros((self.n_machines,self.maxJobs,2), dtype=int)
        self.order = np.zeros((self.n_machines), dtype=int)
        
        return self.state
    
    def step(self, jobToChoose):

        job_times = self.state['job_times']
        job_early_start_time = self.state['job_early_start_time']
        precedence = self.state['precedence']
        machine_utilization = self.state['machine_utilization']

        # Operation index
        jobOrder = self.jobsState[jobToChoose]

        # If job was already finished, raise an exception.
        if jobOrder ==  self.maxJobLength:  
            raise Exception("Job already finished.")

        # Current truncated makespan
        maxSpan = self.maxSpan 
              
        job_early_start =  job_early_start_time[jobToChoose] # how long does it take to start the operations for the chosen job

        job_time =  job_times[jobToChoose][jobOrder] # operation time for the chosen job
        job_machine_placement =  precedence[jobToChoose][jobOrder] # which machine to perform the operation for the chosen job
        
        
        self.placement[job_machine_placement,self.order[job_machine_placement],:] =([jobToChoose, jobOrder])
        self.order[job_machine_placement] += 1

        machine_current_utilization = machine_utilization[job_machine_placement] # how long this machine has been running

        if job_early_start > machine_current_utilization:
            machine_current_utilization = job_early_start + job_time
        else:
            machine_current_utilization += job_time

        job_early_start_time[jobToChoose] = machine_current_utilization   

        machine_utilization[job_machine_placement] = machine_current_utilization
        
        if machine_utilization[job_machine_placement] > maxSpan:
            maxSpan = machine_utilization[job_machine_placement]

        self.jobsState[jobToChoose]+=1
        if  jobOrder+1 == self.maxJobLength:
            self.jobsToProcess -= 1
            self.jobDone[jobToChoose] = 1 
            
        self.reward = float(self.maxSpan - maxSpan)

        # Done
        if  self.jobsToProcess == 0:
            self.done = True

        # Update internal variables
        self.maxSpan = maxSpan
        self.state['job_early_start_time'] = job_early_start_time
        self.state['machine_utilization'] = machine_utilization

        return self.state, self.reward, self.done
    
    def render(self,mode='console'):
        if mode != 'console':
            raise NotImplementedError()
    
    def close(self):
        pass
        