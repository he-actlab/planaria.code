import time
import numpy as np
np.random.seed(111)

class WorkloadGenerator(object):
    def __init__(self, priority_range=range(1,13), distribution='uniform', args=None):
        self.task_types = []
        self.args = args
        self.priority_range = priority_range
        self.distribution = distribution
        self.history = []
    
    def add_task_type(self, name):
        # create workload
        task_type = name

        # add to workloads
        self.task_types.append(task_type)

    def get_last_workload(self):
        assert len(self.history) == 0, 'no workload generated yet'

        return self.history[-1]

    def generate(self):
        num_task = self.args['N']
        generated_workload = []

        # sample numbers prior to the loop if 'poisson'
        if self.distribution == 'poisson':
            poisson_times = np.random.poisson(self.args['arrival_time'], num_task)

        for i in range(num_task):
            # sample cycle at which workload arrives
            if len(generated_workload) == 0:
                cycle = 0
            else:
                if self.distribution == 'uniform':
                    cycle = np.random.randint(self.args['max_cycles'])
                elif self.distribution == 'poisson':
                    incremented_time = poisson_times[i]
                    incremented_cycle = incremented_time * self.args['frequency'] / 1e3
                    previous_cycle = generated_workload[-1][0]
                    cycle = int(previous_cycle + incremented_cycle)
                else:
                    assert 'undefined distribution'

            # random selection of workload
            task_type = np.random.choice(self.task_types)
            
            # random selection of priority
            priority = np.random.choice(self.priority_range)

            # output workload
            generated_task = (cycle, task_type, priority)
            generated_workload.append(generated_task)

        generated_workload.sort(key=lambda x: x[0], reverse=False)
        self.history.append(generated_workload)

        print(generated_workload)
        return generated_workload
