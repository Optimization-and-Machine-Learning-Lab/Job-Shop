from collections import OrderedDict
import numpy as np
import torch


testsize = 1000

jobs = 10
macs = 10
ops = macs
maxTime = 100
precedence = []
time_pre = []


for d in range(testsize):
    np.random.seed(d+2000)
    precedence.append([np.random.choice(ops,ops,replace=False).tolist() for i in range(jobs)])
    time_pre.append(np.random.randint(1,maxTime,size=(jobs,ops)))


precedence = np.array(precedence)
time_pre = np.array(time_pre)

JS_testdata = OrderedDict()

JS_testdata ={
    'testsize': testsize,
    'jobs': jobs,
    'ops': ops,
    'macs': macs,
    'precedence': precedence,
    'time_pre': time_pre
}

torch.save(JS_testdata,'./data/JS_test_%d_%dx%d_t%d.tar'%(testsize,jobs,macs,maxTime))