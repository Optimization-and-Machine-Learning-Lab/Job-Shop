# naive (serial) implementation of multiple environment instances

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

        # mini-batch transition related
        self.BState = []
        self.BSreward = []
        self.BSdone = []
        self.BSinfo = []
        self.BSjobstate = []
        self.tracker = []
        self.multidone = False

        for s in range(self.size):
            self.envs.append(JobShopGymEnv(self.maxJobs,self.maxJobLength,self.n_machines))

    def setGame(self,Precedence,Time_pre):
        for s in range(self.size):
            self.envs[s].setGame(Precedence[s],Time_pre[s])

    def reset(self,BSind):
        self.BState = []
        self.BSreward = []
        self.BSmaxSpan = []
        self.BSdone = []
        self.BSinfo = []
        self.BSjobstate = []
        self.BSmachinestate = []
        self.tracker = []
        for si in range(len(BSind)):
            _state = self.envs[BSind[si]].reset()
            self.BState.append(_state)
            self.BSjobstate.append(self.envs[BSind[si]].jobsState)
            self.BSmachinestate.append(self.envs[BSind[si]].order)
            self.BSmaxSpan.append(self.envs[BSind[si]].maxSpan)
        self.multidone = False

    def faststep(self,BSind_update,JOBTOCHOOSE): 
        self.BState = []
        self.BSreward = []
        self.BSdone = []
        self.BSinfo = []
        self.BSjobstate = []
        self.BSmachinestate = []
        self.BSmaxSpan = []

        for si in range(len(BSind_update)):
            if BSind_update[si] in self.tracker:
                state = -1
                reward = np.nan
                done = True
                info = {}
            else:
                state, reward, done, info = self.envs[BSind_update[si]].faststep(JOBTOCHOOSE[si])
                if done:
                    self.tracker.append(BSind_update[si])

            self.BState.append(state)
            self.BSreward.append(reward)
            self.BSmaxSpan.append(self.envs[BSind_update[si]].maxSpan)
            self.BSdone.append(done)
            self.BSinfo.append(info)
            self.BSjobstate.append(self.envs[BSind_update[si]].jobsState)
            self.BSmachinestate.append(self.envs[BSind_update[si]].order)
            
        if len(self.BSdone)==np.sum(self.BSdone):
            self.multidone = True


