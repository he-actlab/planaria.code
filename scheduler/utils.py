import pandas as pd
import math
import numpy as np
import random
import pickle
import copy
import configparser
import ast
from collections import OrderedDict



def get_cmx_nn_info(path, num_total_cores):
    '''
    This function gets the resulst of the profiling step for all the networks in csv format
    and generates a dictionary with this format: tasks['16 cores']['AlexNet]= [(conv1, 5, 5), ...]
    '''
    # initialize dict
    tasks = OrderedDict({})

    for i in reversed(range(1, num_total_cores+1)):
        # read csv
        csv = pd.read_csv(path + '{}.csv'.format(i)) 

        tasks['{} Cores'.format(i)] = OrderedDict({})

        # get network names
        network_names = csv['Network'].unique()
        
        # iterate network names
        for name in network_names:
            # print(f"processing \"{name}\"")
            
            # initialize list of jobs
            tasks['{} Cores'.format(i)][name] = []

            # filter csv for network name
            csv_network = csv.loc[csv['Network'] == name]

            # translate to atomic jobs within the task
            # here network (task) = N layers & layer = M jobs
            csv_network_layers = list(csv_network['Layer'])
            # csv_network_layers.sort()

            for layer_name in csv_network_layers:
                layer = csv_network.loc[csv['Layer'] == layer_name]
                layer_cycle = np.array(layer['Total Cycles'])[0]
                layer_tiles = np.array(layer['Tiling'])[0]
                layer_energy = np.array(layer['Energy'])[0]
                
                job_cycle = math.floor(layer_cycle / layer_tiles)

                # insert into tasks[name] 
                tasks['{} Cores'.format(i)][name].append((layer_name, layer_tiles, job_cycle, layer_energy))

    return tasks


def save_data(data, data_file_name):
    with open(data_file_name, "wb") as f:
        pickle.dump(data, f)



def read_csv(file_name):
    
    # initialize dict
    tasks = {}

    # read csv
    csv = pd.read_csv(file_name)

    # get network names
    network_names = csv['Network'].unique()
    
    # iterate network names
    for name in network_names:
        print(f"processing \"{name}\"")
        
        # initialize list of jobs
        tasks[name] = []

        # filter csv for network name
        csv_network = csv.loc[csv['Network'] == name]

        # translate to atomic jobs within the task
        # here network (task) = N layers & layer = M jobs
        csv_network_layers = csv_network['Layer']
        for layer_name in csv_network_layers:
            layer = csv_network.loc[csv['Layer'] == layer_name]
            layer_cycle = np.array(layer['Cycles'])[0]
            layer_tiles = np.array(layer['Tiling'])[0]
            
            job_cycle = math.floor(layer_cycle / layer_tiles)

            # insert into tasks[name]
            new_jobs = [(layer_name, job_cycle)] * layer_tiles
            tasks[name].extend(new_jobs)


    return tasks





