from collections import OrderedDict
import numpy as np
import torch
from utils import *

train_problem_sizes = [[6,6], [10,10], [15,15], [20,15], [20,20], [30,15], [30,20], [50,15], [50,20], [100,20]] 

def create_test(jobs, macs, testsize):

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


def create_test_ta():

    i = 0
    for size in train_problem_sizes[2:]:
        jobs, macs = size
        ops = macs
        maxTime = 100
        testsize = 10
        precedence = []
        time_pre = []

        for d in range(1, testsize + 1):
            test_precedence,test_timepre_ = read_instances('./data/ta/ta{}'.format(i*10 + d))
            precedence.append(test_precedence[0])
            time_pre.append(test_timepre_[0])

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
        i += 1

create_test_ta()
# create_test(10, 10, 10)