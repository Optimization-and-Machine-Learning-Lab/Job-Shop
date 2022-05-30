import numpy as np

def read_instances(file_test):
    
    testing_machines = []
    testing_duration = []

    # Testing dataset

    num_lines = len(open(file_test).readlines())

    with open(file_test, "r") as file:

        NB_JOBS, NB_MACHINES = [int(v) for v in file.readline().split()]

        for _ in range(num_lines // (NB_JOBS)):

            JOBS = [[int(v) for v in file.readline().split()] for i in range(NB_JOBS)]

            # Build list of machines. MACHINES[j][s] = id of the machine for the operation s of the job j
            testing_machines.append([[JOBS[j][2 * s] for s in range(NB_MACHINES)] for j in range(NB_JOBS)])

            # Build list of durations. DURATION[j][s] = duration of the operation s of the job j
            testing_duration.append([[JOBS[j][2 * s + 1] for s in range(NB_MACHINES)] for j in range(NB_JOBS)])


    testing_machines = np.array(testing_machines)
    testing_duration = np.array(testing_duration)
    
    return testing_machines, testing_duration