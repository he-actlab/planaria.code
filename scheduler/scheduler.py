import pandas as pd
import math
import numpy as np
import random
import pickle
import copy
import configparser
import ast
from collections import OrderedDict



class TaskStats(object):
    """
    This class stores the stats for the tasks in the queue
    """

    def __init__(self, task_name, start_time, current_time, priority, SLA, num_cores_executing, progress, nn_info):
        
        self.task_name = task_name
        self.start_time = start_time
        self.current_time = current_time
        self.priority = priority
        self.sla = SLA
        self.num_cores_executing = num_cores_executing
        self.energy = 0

        ##### progress = (layer, % of that layer executed)
        self.progress = progress
        
        task_info = OrderedDict({})
        for core in nn_info.keys():
            task_info[core] = nn_info[core][task_name]
        self.info = task_info


    def get_remaining_estimated_time(self, num_cores):
        """
        This fucntion calculates the remaining estimated time to execute the NN with a specific number of subarrays (cores)
        """
        if num_cores > 0:
            nn_stats = self.info['{} Cores'.format(int(num_cores))]
            current_layer, progress = self.progress
            estimated_time = 0

            for idx, layer_stats in enumerate(reversed(nn_stats)):
                
                if layer_stats[0] == current_layer:
                    ##### We need to estimate the remaining jobs of the current layer
                    layer_total_cycles = layer_stats[1] * layer_stats[2]
                    layer_remaining_cycles = self.get_time_to_finish_current_layer(num_cores)
                    estimated_time += layer_remaining_cycles
                    break  
                else:
                    estimated_time += layer_stats[1] * layer_stats[2]            
        else:
            estimated_time = float('inf')
        
        return estimated_time
    

    def get_time_to_finish_current_layer(self, num_cores):
        """
        This function returns the number of cycles required to finish the current layer with a number of subarrays (cores)
        """
        nn_stats = self.info['{} Cores'.format(int(num_cores))]
        current_layer_name, progress = self.progress
        current_layer_stats = list(filter(lambda x: x[0] == current_layer_name, nn_stats))[0]
        current_layer_num_tiles = current_layer_stats[1]
        current_layer_num_cycles_per_tile = current_layer_stats[2]

        num_tiles_to_finish_layer = math.ceil((1 - progress) * current_layer_num_tiles)
        time_to_finish = num_tiles_to_finish_layer * current_layer_num_cycles_per_tile

        return time_to_finish
    

    def get_task_slowdown(self, num_total_cores):
        nn_stats = self.info['{} Cores'.format(int(num_total_cores))]
        current_layer_name, progress = self.progress
        current_layer_stats = list(filter(lambda x: x[0] == current_layer_name, nn_stats))[0]
        current_layer_num_tiles = current_layer_stats[1]
        current_layer_num_cycles_per_tile = current_layer_stats[2]

        layer_index = nn_stats.index(current_layer_stats)
        layer_list = nn_stats[0:layer_index]
        isolated_time_taken = sum(ls[1] * ls[2] for ls in layer_list) + progress * current_layer_num_cycles_per_tile * current_layer_num_tiles
        multi_time_taken = self.current_time - self.start_time

        if isolated_time_taken != 0:
            slowdown = multi_time_taken/float(isolated_time_taken)
        else:
            slowdown = 1

        return slowdown



    def get_next_scheduling_time(self, next_raw_scheduling_time, num_cores):
        """
        Based on the next raw data for scheduling, this function sees when the task will come to/out queue and if it needs to finish its current tile and postpone scheduling or not
        It returns the proposed scheduling time (the time for task reallocation after it finishes its current layer), next state(layer) and its progress
        """
        proposed_scheduling_time = 0
        if num_cores > 0:
            nn_stats = self.info['{} Cores'.format(int(num_cores))]
            current_layer, progress = self.progress
            current_layer_stats = list(filter(lambda x: x[0] == current_layer, nn_stats))[0]
            
            time_to_finish_current_layer = self.get_time_to_finish_current_layer(num_cores)

            if next_raw_scheduling_time <= time_to_finish_current_layer:
                num_tiles = math.ceil(next_raw_scheduling_time / float(current_layer_stats[2]))
                proposed_scheduling_time = num_tiles * current_layer_stats[2]
                new_progress = progress + num_tiles/float(current_layer_stats[1])
                new_layer = current_layer
                result = [proposed_scheduling_time, new_layer, new_progress]
                
            else:
                proposed_scheduling_time += time_to_finish_current_layer
                for layer_idx in range(nn_stats.index(current_layer_stats)+1, len(nn_stats)):
                    layer_time = nn_stats[layer_idx][1] * nn_stats[layer_idx][2]
                    remaining_time = next_raw_scheduling_time - proposed_scheduling_time
                    if layer_time < remaining_time:
                        proposed_scheduling_time += layer_time
                        
                    elif layer_time == remaining_time:
                        proposed_scheduling_time += layer_time
                        if layer_idx == len(nn_stats) - 1:
                            new_layer = nn_stats[layer_idx][0]
                            new_progress = 1
                        else:
                            new_layer = nn_stats[layer_idx+1][0]
                            new_progress = 0
                        result = [proposed_scheduling_time, new_layer, new_progress]
                        break

                    else:
                        num_tiles = math.ceil(remaining_time / float(nn_stats[layer_idx][2]))
                        final_time = num_tiles * nn_stats[layer_idx][2]

                        if final_time == layer_time:
                            proposed_scheduling_time += layer_time
                            if layer_idx < len(nn_stats) - 1:

                                new_layer = nn_stats[layer_idx+1][0]
                                new_progress = 0
                            else:
                                new_layer = nn_stats[layer_idx][0]
                                new_progress = 1
                            result = [proposed_scheduling_time, new_layer, new_progress]
                            break
                        else:
                            assert final_time < layer_time, "We cannot finish this layer!"
                            proposed_scheduling_time += final_time
                            new_layer = nn_stats[layer_idx][0]
                            new_progress = float(num_tiles) / float(nn_stats[layer_idx][1])
                            result = [proposed_scheduling_time, new_layer, new_progress]
                            break
        else:
            current_layer, progress = self.progress
            result = [next_raw_scheduling_time, current_layer, progress]

        return result


    def update_task(self, task_proposed_scheduling_info, decided_scheduling_time, num_cores):
        """
        Based on the decided next scheduling time, this function updates the task attributes and their progress
        """
        energy_consumption = 0
        if num_cores > 0:
            nn_stats = self.info['{} Cores'.format(int(num_cores))]
            current_layer, progress = self.progress
            current_layer_stats = list(filter(lambda x: x[0] == current_layer, nn_stats))[0]
            ##### updating energy
            energy_consumption += (1-progress) * current_layer_stats[3]
            next_layer = task_proposed_scheduling_info[1]
            next_progress = task_proposed_scheduling_info[2]
            next_layer_stats = list(filter(lambda x: x[0] == next_layer, nn_stats))[0]
            energy_consumption += next_progress * next_layer_stats[3]

            current_layer_idx = nn_stats.index(current_layer_stats)
            next_layer_idx = nn_stats.index(next_layer_stats)

            for idx in range(current_layer_idx+1, next_layer_idx):
                energy_consumption += nn_stats[idx][3]
        else:
            pass
            
        self.energy += energy_consumption

        self.current_time += decided_scheduling_time
        self.num_cores_executing = num_cores
        
        task_proposed_scheduling_time = task_proposed_scheduling_info[0]
        task_next_layer = task_proposed_scheduling_info[1]
        task_next_progress = task_proposed_scheduling_info[2]



        assert task_proposed_scheduling_time <= decided_scheduling_time, "The decided and proposed scheduling time comply when we update the tasks"

        self.progress = (task_next_layer, task_next_progress)

        return



##### Planaria spatial multi-tenant scheduler
def scheduler(workload, nn_info, QoS_dict, frequency, num_total_cores):
    '''
    This is the main scheduler fucntion that receives the workload and schedules it and returns the stats
    '''
    task_arrivals = OrderedDict({})
    for idx, t in enumerate(workload):
        task_arrivals[t[1]+'{}'.format(idx)] = t[0]
        print(t[1]+'{}'.format(idx))
    
    task_arrivals_original = copy.deepcopy(task_arrivals)

    done_tasks_list =[]
    task_queue = {}
    period = 175*1e300
    cycle = 0

    TASK_FINISHES = False
    PERIOD_REACHED = False
    ##### We add the first task, at cycle = 0
    TASK_ARRIVES = True
    task_queue[workload[0][1]] = TaskStats(workload[0][1], cycle, cycle, workload[0][2],\
                                        QoS_dict[workload[0][1]], 0, (nn_info['16 Cores'][workload[0][1]][0][0], 0), nn_info)
    del task_arrivals[list(task_arrivals.keys())[0]]

    print('task arrivals', task_arrivals)
    print('{} Arrives'.format(task_queue[workload[0][1]].task_name) )


    ##### We keep scheduling until all the tasks are finished
    period_counter = 0
    while len(done_tasks_list) < len(workload):

        #### At these curcumestances we need to update the schedule
        if TASK_ARRIVES or TASK_FINISHES or PERIOD_REACHED:

            print('Number of finished tasks', len(done_tasks_list))
            possible_num_cores_dict = {}

            for key, task in task_queue.items():
                task_possible_num_cores = get_possible_num_cores(task, frequency, num_total_cores)
                possible_num_cores_dict[key] = task_possible_num_cores

            if check_if_tasks_all_fit(possible_num_cores_dict, num_total_cores):
                ##### Now we need to assign the cores to the tasks
                tasks_cores = assign_cores_if_tasks_fit(task_queue, possible_num_cores_dict, num_total_cores)  

            else:
                tasks_cores = assign_cores_if_tasks_not_fit(task_queue, possible_num_cores_dict, num_total_cores, frequency)

            ##### Now we need to figure out what is the next scheduling time
            ##### First we see when the tasks are gonna be finished
            if len(list(task_queue.keys())) > 0:

                task_estimated_finish_time = {}
                for key, task in task_queue.items():
                    task_est_finish = task.get_remaining_estimated_time(tasks_cores[key])
                    task_estimated_finish_time[key] = task_est_finish

                ##### we sort the task finish time from min to max to find the closest one
                task_estimated_finish_time_sorted = sorted(task_estimated_finish_time.items(), key=lambda kv: kv[1], reverse=False)

                ##### Next task to finish
                next_task_to_finish_name, next_task_to_finish_time = task_estimated_finish_time_sorted[0]
                ##### Next period arrival
                next_period_time = (period_counter + 1) * period - cycle
                ##### Next task arrival
                if len(task_arrivals) > 0:
                    next_task_arrival_time = task_arrivals[list(task_arrivals.keys())[0]]
                    next_task_arrival_time -= cycle
                else:
                    next_task_arrival_time = None
                

                ##### Now we update the cycles of tasks and set the next scheduling time
                if next_task_arrival_time is not None:
                    next_scheduling_time = min(next_task_to_finish_time, next_period_time, next_task_arrival_time)


                    proposed_scheduling_time_dict = {}
                    proposed_scheduling_time_list = []
                    for key, task in task_queue.items():
                        r = task.get_next_scheduling_time(next_scheduling_time, tasks_cores[key])
                        proposed_scheduling_time_dict[key] = r
                        proposed_scheduling_time_list.append(r[0])

                    final_proposed_scheduling_time = max(proposed_scheduling_time_list)

                    if next_scheduling_time == next_task_to_finish_time:
                        ##### The next scheduling time will be when a task finishes
                        assert final_proposed_scheduling_time < next_task_arrival_time and final_proposed_scheduling_time < next_period_time, \
                                                            "Due to the tile asynchronity, the next scheduling policy has changed! Task should finish" 
                        
                        ##### Now we need to update the tasks based on the proposed scheduling time
                        for key, task in task_queue.items():
                            task.update_task(proposed_scheduling_time_dict[key], final_proposed_scheduling_time, tasks_cores[key])

                        ##### We update the global scheduler
                        print('{} finished'.format(next_task_to_finish_name))
                        cycle += final_proposed_scheduling_time
                        print("task doen", cycle)
                        done_tasks_list.append(task_queue[next_task_to_finish_name])
                        del task_queue[next_task_to_finish_name]
                        TASK_FINISHES = True
                        TASK_ARRIVES = False
                        PERIOD_REACHED = False
                        
                    
                    elif next_scheduling_time == next_period_time:
                        ##### The next scheduling time will be when the next period reached

                        ##### Now we need to update the tasks based on the proposed scheduling time
                        for key, task in task_queue.items():
                            task.update_task(proposed_scheduling_time_dict[key], final_proposed_scheduling_time, tasks_cores[key])
                        
                        ##### We update the global scheduler
                        print('Next period reached')
                        cycle += final_proposed_scheduling_time
                        period_counter += 1
                        PERIOD_REACHED = True
                        TASK_FINISHES = False
                        TASK_ARRIVES = False
                    
                    elif next_scheduling_time == next_task_arrival_time:
                        ##### The next scheduling time will be when the next task arrives

                        assert final_proposed_scheduling_time < next_task_to_finish_time and final_proposed_scheduling_time < next_period_time, \
                                                            "Due to the tile asynchronity, the next scheduling policy has changed! Task should finish" 
                        ##### Now we need to update the tasks based on the proposed scheduling time
                        for key, task in task_queue.items():
                            task.update_task(proposed_scheduling_time_dict[key], final_proposed_scheduling_time, tasks_cores[key])   
                                    
                        ##### We update the global scheduler
                        name = list(task_arrivals.keys())[0]
                        idx = list(task_arrivals_original.keys()).index(name)
                        task_new = workload[idx]


                        task_queue[name] = TaskStats(workload[idx][1], next_task_arrival_time + cycle, cycle + final_proposed_scheduling_time, workload[idx][2],\
                                                                QoS_dict[workload[idx][1]], 0, (nn_info['16 Cores'][workload[idx][1]][0][0], 0), nn_info)
                        del task_arrivals[name]
                        print('task arrivals', task_arrivals)
                        print('{} Arrives'.format(workload[idx][1]), task_queue[name].start_time, task_queue[name].current_time)                
                        cycle += final_proposed_scheduling_time
                        TASK_ARRIVES = True
                        TASK_FINISHES = False
                        PERIOD_REACHED = False
                else:
                    ##### All the tasks have arrived. Next scheduling time will be either period reached or task finishes
                    next_scheduling_time = min(next_task_to_finish_time, next_period_time)

                    proposed_scheduling_time_dict = {}
                    proposed_scheduling_time_list = []
                    for key, task in task_queue.items():
                        r = task.get_next_scheduling_time(next_scheduling_time, tasks_cores[key])
                        proposed_scheduling_time_dict[key] = r
                        proposed_scheduling_time_list.append(r[0])

                    final_proposed_scheduling_time = max(proposed_scheduling_time_list)

                    if next_scheduling_time == next_task_to_finish_time:
                        ##### The next scheduling time will be when a task finishes
                        ##### Now we need to update the tasks based on the proposed scheduling time
                        for key, task in task_queue.items():
                            task.update_task(proposed_scheduling_time_dict[key], final_proposed_scheduling_time, tasks_cores[key])

                        ##### We update the global scheduler

                        print('{} finished'.format(next_task_to_finish_name))
                        cycle += final_proposed_scheduling_time
                        print("task done", cycle)
                        done_tasks_list.append(task_queue[next_task_to_finish_name])
                        del task_queue[next_task_to_finish_name]
                        TASK_FINISHES = True
                        TASK_ARRIVES = False
                        PERIOD_REACHED = False
                        
                    
                    elif next_scheduling_time == next_period_time:
                        ##### The next scheduling time will be when the next period reached

                        ##### Now we need to update the tasks based on the proposed scheduling time
                        for key, task in task_queue.items():
                            task.update_task(proposed_scheduling_time_dict[key], final_proposed_scheduling_time, tasks_cores[key])
                        
                        ##### We update the global scheduler
                        print('Next period reached')
                        cycle += final_proposed_scheduling_time
                        period_counter += 1
                        PERIOD_REACHED = True
                        TASK_FINISHES = False
                        TASK_ARRIVES = False

            else:
                ##### Task Queue is empty, next scheduling will be task arrival or period reached!
                ##### Next period arrival
                next_period_time = (period_counter + 1) * period - cycle
                ##### Next task arrival
                if len(list(task_arrivals.keys()))>0:
                    next_task_arrival_time = task_arrivals[list(task_arrivals.keys())[0]]
                else:
                    next_task_arrival_time = None

                ##### Now we update the cycles of tasks and set the next scheduling time
                if next_task_arrival_time is not None:
                    next_scheduling_time = min(next_period_time, next_task_arrival_time)
                        
                    if next_scheduling_time == next_period_time:
                        ##### The next scheduling time will be when the next period reached
                        
                        ##### We update the global scheduler
                        print('Next period reached')
                        cycle += final_proposed_scheduling_time
                        period_counter += 1
                        PERIOD_REACHED = True
                        TASK_FINISHES = False
                        TASK_ARRIVES = False
                    
                    elif next_scheduling_time == next_task_arrival_time:
                        ##### The next scheduling time will be when the next task arrives
                                    
                        ##### We update the global scheduler
                        name = list(task_arrivals.keys())[0]
                        idx = list(task_arrivals_original.keys()).index(name)
                        task_new = workload[idx]

                        task_queue[name] = TaskStats(workload[idx][1], next_task_arrival_time + cycle, cycle + next_task_arrival_time, workload[idx][2],\
                                                                QoS_dict[workload[idx][1]], 0, (nn_info['16 Cores'][workload[idx][1]][0][0], 0), nn_info)
                        del task_arrivals[name]
                        print('{} Arrives'.format(workload[idx][1]), task_queue[name].start_time, task_queue[name].current_time)               
                        cycle += final_proposed_scheduling_time
                        TASK_ARRIVES = True
                        TASK_FINISHES = False
                        PERIOD_REACHED = False

    
    return done_tasks_list







def check_if_tasks_all_fit(possible_num_cores_dict, num_total_cores):
    """
    This fuction check to see if all the tasks can be mapped simulatanously on the accelerator while meeting QoS
    """
    ##### We check the minimum possible number of cores
    possible_num_cores_list = list(possible_num_cores_dict.values())
    sum_p_c = 0
    for p_c in possible_num_cores_list:
        if len(p_c) > 0:
            sum_p_c += p_c[-1]
        else:
            return False
    if sum_p_c <= num_total_cores:
        return True
    else:
        return False






def assign_cores_if_tasks_fit(task_queue, possible_num_cores_dict, num_total_cores):
    """
    This function assigns the subarrays (cores) to nns in queue when they can all fit on the accelerator and meets their QoS
    It returns a dictionary containing the number of cores (subarrays) for each task
    """
    tasks_cores_dict = {}
    if len(list(task_queue.keys())) > 0:

        ##### We first assign minimum number of cores that tasks meet their QoS
        for key, task in task_queue.items():
            tasks_cores_dict[key] = possible_num_cores_dict[key][-1]
        available_spare_cores = num_total_cores - sum(list(tasks_cores_dict.values()))

        ##### We then assign the spare cores to the tasks based on their priority.
        total_priority = sum(task.priority for key,task in task_queue.items())

        task_scores = {}
        for key, task in task_queue.items():
            if task.get_remaining_estimated_time(possible_num_cores_dict[key][-1]) == 0:
                task_score = 0
            else:
                task_score = (task.priority) * 1 / float(task.get_remaining_estimated_time(possible_num_cores_dict[key][-1]))
            task_scores[key] = task_score
        

        sorted_task_scores = sorted(task_scores.items(), key=lambda kv: kv[1], reverse=True)
        sum_scores = sum(list(task_scores.values()))

        num_available_cores = available_spare_cores
        for t_s in sorted_task_scores:
            task_additional_core = round((t_s[1] /float(sum_scores)) * num_available_cores)
            tasks_cores_dict[t_s[0]] += task_additional_core


        ##### Check to see if we are overutilizing/underutilizing the cores
        utilized_cores = sum(list(tasks_cores_dict.values()))

        if utilized_cores == num_total_cores:
            return tasks_cores_dict
        
        elif utilized_cores < num_total_cores:
            final_spare_cores = num_total_cores - utilized_cores
            ##### We allocate the final spare cores to the task with highest priority
            ##### Finding the task with maximum priority
            tasks_with_max_priority = find_tasks_with_max_priority(task_queue)
            for t_max_p in tasks_with_max_priority:
                tasks_cores_dict[t_max_p] += math.floor(final_spare_cores / float(len(tasks_with_max_priority)))

            if math.floor(final_spare_cores / float(len(tasks_with_max_priority))) * len(tasks_with_max_priority) < final_spare_cores:
                ##### selecting a task randomly and give extra cores to it
                lucky_task = random.choice(tasks_with_max_priority)
                tasks_cores_dict[lucky_task] += final_spare_cores - math.floor(final_spare_cores / float(len(tasks_with_max_priority))) * len(tasks_with_max_priority)
                
            return tasks_cores_dict

    
        elif utilized_cores > num_total_cores:
            overutilized_cores = utilized_cores - num_total_cores

            ##### We first check if in low priority tasks there is a candidate to take cores from it and still meet its SLA
            tasks_sorted_from_min_to_max_priority = sort_tasks_from_min_to_max_priority(task_queue)
            for t_min_p in tasks_sorted_from_min_to_max_priority:
                if tasks_cores_dict[t_min_p[0]] - 1 >= possible_num_cores_dict[t_min_p[0]][-1]:
                    tasks_cores_dict[t_min_p[0]] -= 1
                
                if sum(list(tasks_cores_dict.values())) == num_total_cores:
                    break
                else:
                    pass
            
            assert sum(list(tasks_cores_dict.values())) == num_total_cores, "We are overutilizing the cores"

            return tasks_cores_dict
    else:
        return tasks_cores_dict


def assign_cores_if_tasks_not_fit(task_queue, possible_num_cores_dict, num_total_cores, frequency):
    """
    This function assigns cores (subarrays) to the tasks when they do not all fit on the accelerator (due to QoS violation) and there is a race between them
    It returns a dictionary of the number of cores for each tasks
    """
    tasks_cores_dict = {}
    ##### We first assign scores to the tasks in queue and sort them based on that
    task_scores = {}
    task_not_meet_sla = []

    for key, task in task_queue.items():
        ##### First calculate the slack for each task
        sla_in_cycle = task.sla * 1.e-3 * frequency
        task_executed_time = task.current_time - task.start_time
        slack = sla_in_cycle - task_executed_time
        ##### Now we assign scores 
        if len(possible_num_cores_dict[key]) > 0:
            if task.get_remaining_estimated_time == 0:
                task_score = 0
            else:
                task_score = task.priority * (1 / float(slack)) * (1 / float(possible_num_cores_dict[key][-1]))

            task_scores[key] = task_score
        else:
            if task.get_remaining_estimated_time(num_total_cores) > 0:
                task_score = task.priority * (-1) * 1 / float(task.get_remaining_estimated_time(num_total_cores))
            else:
                task_score = task.priority * (-1) * 1 / 1
            task_scores[key] = task_score
            task_not_meet_sla.append(key)
    
    ###### Now we need to sort the scores from max to min
    sorted_task_scores = sorted(task_scores.items(), key=lambda kv: kv[1], reverse=True)

    ##### check to see if the tasks that they have not meet their SLAs only exist
    if len(task_not_meet_sla) < len(list(task_queue.keys())):

        num_available_cores = num_total_cores
        for t_s in sorted_task_scores:
            if len(possible_num_cores_dict[t_s[0]]) > 0:
                cores_to_assign = possible_num_cores_dict[t_s[0]][-1]
                
            else:
                cores_to_assign = 0
            if num_available_cores - cores_to_assign >= 0:
                tasks_cores_dict[t_s[0]] = cores_to_assign
                num_available_cores -= cores_to_assign
            else:
                if num_available_cores > 0:
                    tasks_cores_dict[t_s[0]] = num_available_cores
                    num_available_cores = 0
                else:
                    tasks_cores_dict[t_s[0]] = 0

        available_spare_cores = num_total_cores - sum(list(tasks_cores_dict.values()))
        sum_scores = sum(t_s[1] for t_s in sorted_task_scores if t_s[1]>=0)

        num_available_cores = available_spare_cores
        for t_s in sorted_task_scores:
            if t_s[1] >= 0:
                task_additional_core = round((t_s[1] /float(sum_scores)) * num_available_cores)
            else:
                task_additional_core = 0
            tasks_cores_dict[t_s[0]] += task_additional_core

        
        ##### Check to see if we are overutilizing/underutilizing the cores
        utilized_cores = sum(list(tasks_cores_dict.values()))

        if utilized_cores == num_total_cores:
            return tasks_cores_dict
        
        elif utilized_cores < num_total_cores:
            final_spare_cores = num_total_cores - utilized_cores
            ##### We allocate the final spare cores to the task with highest priority
            ##### Finding the task with maximum priority
            tasks_with_max_priority = find_tasks_with_max_priority(task_queue)
            for t_max_p in tasks_with_max_priority:
                tasks_cores_dict[t_max_p] += math.floor(final_spare_cores / float(len(tasks_with_max_priority)))

            if math.floor(final_spare_cores / float(len(tasks_with_max_priority))) * len(tasks_with_max_priority) < final_spare_cores:
                ##### selecting a task randomly and give extra cores to it
                lucky_task = random.choice(tasks_with_max_priority)
                tasks_cores_dict[lucky_task] += final_spare_cores - math.floor(final_spare_cores / float(len(tasks_with_max_priority))) * len(tasks_with_max_priority)
                
            return tasks_cores_dict

    
        elif utilized_cores > num_total_cores:
            overutilized_cores = utilized_cores - num_total_cores

            ##### We first check if in low priority tasks there is a candidate to take cores from it and still meet its SLA
            tasks_sorted_from_min_to_max_priority = sort_tasks_from_min_to_max_priority(task_queue)
            for t_min_p in tasks_sorted_from_min_to_max_priority:
                if len(possible_num_cores_dict[t_min_p[0]]) >0:

                    if tasks_cores_dict[t_min_p[0]] - 1 >= possible_num_cores_dict[t_min_p[0]][-1]:
                        tasks_cores_dict[t_min_p[0]] -= 1
                else:
                    pass
                
                if sum(list(tasks_cores_dict.values())) == num_total_cores:
                    break
                else:
                    pass
            
            assert sum(list(tasks_cores_dict.values())) == num_total_cores, "We are overutilizing the cores"
        
        return tasks_cores_dict

    elif len(task_not_meet_sla) == len(list(task_queue.keys())):
        num_available_cores = num_total_cores
        sorted_task_scores = sorted(task_scores.items(), key=lambda kv: kv[1], reverse=False)
        total_task_scores = abs(sum(t[1] for t in sorted_task_scores))

        for t_s in sorted_task_scores:
            cores_to_assign = round(abs(t_s[1]/float(total_task_scores)) * num_total_cores)
            if num_available_cores - cores_to_assign >= 0:
                tasks_cores_dict[t_s[0]] = cores_to_assign
                num_available_cores -= cores_to_assign
            else:
                if num_available_cores > 0:
                    tasks_cores_dict[t_s[0]] = num_available_cores
                    num_available_cores = 0
                else:
                    tasks_cores_dict[t_s[0]] = 0
        return tasks_cores_dict            


def find_tasks_with_max_priority(task_queue):
    """
    This fucntion finds all the tasks in the queue with max priority 
    """

    ##### when we want to add cores, we add it to the maximum priority cores
    tasks_with_max_priority = []
    p = 1
    for key, task in task_queue.items():
        if task.priority > p:
            p = task.priority
            tasks_with_max_priority.clear()
            tasks_with_max_priority.append(key)
        elif task.priority == p:
            tasks_with_max_priority.append(key)
        else:
            pass
    return tasks_with_max_priority


def sort_tasks_from_min_to_max_priority(task_queue):
    """
    Sorting tasks from min to max pririty
    retuerning the list of their names
    """
    #### first generating a dict from the tasks priorities
    tasks_priorities = {}
    for key, task in task_queue.items():
        tasks_priorities[key] = task.priority

    sorted_tasks_priorities = sorted(tasks_priorities.items(), key=lambda kv: kv[1])
    return sorted_tasks_priorities



          

def get_possible_num_cores(task, frequency, num_total_cores):
    """
    This function finds the number of possible cores (subarrays) for a task that still meets its QoS
    """
    sla_in_cycle = task.sla * 1.e-3 * frequency
    task_executed_time = task.current_time - task.start_time
    slack = sla_in_cycle - task_executed_time

    possible_num_cores = []
    ##### caculate the remaining time based on the number of cores
    for core in reversed(range(1, num_total_cores+1)):

        est_time = task.get_remaining_estimated_time(core)
        if est_time <= slack:
            possible_num_cores.append(core)
        else:
            pass

    return possible_num_cores 



