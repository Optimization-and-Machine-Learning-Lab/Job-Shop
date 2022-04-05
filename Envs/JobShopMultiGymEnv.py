import numpy as np
from Envs.JobShopGymEnv import *

class JobShopMultiGymEnv(object):
    def __init__(self, size, maxJobs=10,maxJobLength=10,n_machines=10):
        super(JobShopMultiGymEnv, self).__init__()

        self.size = size
        self.envind = [ind for ind in range(size)]
        self.maxJobs=maxJobs
        self.maxJobLength=maxJobLength
        self.n_machines = n_machines
        self.envs = []

        for _ in range(self.size):
            self.envs.append(JobShopGymEnv(self.maxJobs,self.maxJobLength,self.n_machines))

    def setGame(self,Precedence,Time_pre):
        for s in range(self.size):
            self.envs[s].setGame(Precedence[s],Time_pre[s])

    def reset(self,BSind):
        self.multidone = False
        self.tracker = []

        for si in range(len(BSind)):
            self.envs[BSind[si]].reset()

        self.bstate = self._update_BState(BSind)
        return self.bstate

    def _update_BState(self, BSind):

        mac_utl = np.array([self.envs[BSind[i]].state['machine_utilization'] for i in range(len(BSind))])
        job_time = np.array([self.envs[BSind[i]].state['job_times'] for i in range(len(BSind))])   
        job_early = np.array([self.envs[BSind[i]].state['job_early_start_time'] for i in range(len(BSind))])   
        pre = np.array([self.envs[BSind[i]].state['precedence'] for i in range(len(BSind))])                     

        job_state = np.array([self.envs[BSind[i]].jobsState for i in range(len(BSind))])
        machine_state = np.array([self.envs[BSind[i]].order for i in range(len(BSind))])
        makespan = np.array([self.envs[BSind[i]].maxSpan for i in range(len(BSind))])

        bstate = {
            'machine_utilization': mac_utl,
            'job_times': job_time,
            'job_early_start_time': job_early,
            'precedence': pre,
            'job_state': job_state,
            'machine_state': machine_state,
            'makespan': makespan
        }
        return bstate

    def step(self,BSind_update,JOBTOCHOOSE): 

        for i in range(len(BSind_update)):

            if BSind_update[i] in self.tracker:
                raise Exception("Job already finished.")
           
            state, reward, done = self.envs[BSind_update[i]].step(JOBTOCHOOSE[i])
            if done:
                self.tracker.append(BSind_update[i])

        self.bstate = self._update_BState(BSind_update)
        breward = np.array([self.envs[BSind_update[i]].reward for i in range(len(BSind_update))])
        bdone = np.array([self.envs[BSind_update[i]].done for i in range(len(BSind_update))])

        if len(bdone)==np.sum(bdone):
            self.multidone = True

        return self.bstate, breward, self.multidone
