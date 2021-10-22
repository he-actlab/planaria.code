import pandas as pd
import math
import numpy as np
import random
import pickle
import copy
import configparser
import ast
from collections import OrderedDict

from generator import WorkloadGenerator
from scheduler import scheduler
from utils import get_cmx_nn_info, save_data


def main():
    '''
    Scheduler for Planaria Spatial-Multi-Tenant accelerator
    '''
    ##############################################################
    ##### Getting the scheduling parameters from the config file #####
    ##############################################################
    config_file = 'scheduler.ini'
    config = configparser.ConfigParser()
    config.read('scheduler.ini')

    ### accelerator configs
    frequency = float(config.get('accelerator', 'frequency')) * 1.e6
    num_total_cores = int(config.get('accelerator', 'num_active_subarrays'))

    ### scheduling system configs
    number_of_tasks = int(config.get('system', 'num_tasks'))
    ### poisson or uniform
    arrival_time_distribution = ast.literal_eval(config.get('system', 'arrival_time_distribution'))
    ### poisson parameter
    arrival_time = float(config.get('system', 'arrival_time'))
    ### start time range for uniform distribution
    max_cycles = float(config.get('system', 'max_cycles'))

    ### DNN tasks and their QoS constraints (in ms) for the target workload scenario
    QoS_dict = ast.literal_eval(config.get('workload', 'workload_scenario_qos'))
    task_types = list(QoS_dict.keys())

    ### Path to input CSVs of the Planaria performance numbers for various number of subarrays and their optimized fission configuration
    path_to_input_csv = ast.literal_eval(config.get('csv_path', 'path_to_input_csv_files'))

    ######## Output results file names
    output_file_name = 'scheduled-workload-planaria.data'
    output_csv_file_name = 'scheduled-workload-planaria-stats.csv'   
    ##############################################################
    ##############################################################
    ##############################################################



    ######## Getting the DNN tasks info from csvs
    nn_info = get_cmx_nn_info(path_to_input_csv, num_total_cores)

    ### Isolated execution time of the DNN tasks
    nn_isolated_time_info = nn_info['{} Cores'.format(num_total_cores)]
    nn_isolated_time = {}
    for key, stats in nn_isolated_time_info.items():
        isolated_time_list = nn_isolated_time_info[key]
        isolated_time = sum(x[1]*x[2] for x in isolated_time_list)
        nn_isolated_time[key] = isolated_time
    

    ######## Generating a workload #######
    generator_args = {'max_cycles':max_cycles*0.25,
                    'arrival_time':arrival_time,
                    'frequency':frequency,
                    'N':number_of_tasks}

    generator = WorkloadGenerator(distribution=arrival_time_distribution, args=generator_args)
    for task_type in task_types:
        generator.add_task_type(task_type)
    workload = generator.generate()

    ##############################################################
    ######## Here is the scheduled workload ##############
    ##############################################################
    scheduled_workload = scheduler(workload, nn_info, QoS_dict, frequency, num_total_cores)

    print('*'*50)
    print('Scheduling Done')
    print('*'*50)

    ######## We dump the scheduled workload
    save_data(scheduled_workload, output_file_name)

    ####### Generating the output CSV file
    output_csv_col = ['Start time (ms)', 'Finish time (ms)', 'Total execution time (ms)', 'QoS (ms)', 'Isolated execution time (ms)']
    output_csv_df = pd.DataFrame(columns=output_csv_col)

    for task in scheduled_workload:
        task_stats = [task.start_time/frequency*1000, task.current_time/frequency*1000, (task.current_time - task.start_time)/frequency*1000, task.sla, nn_isolated_time[task.task_name]/frequency*1000]
        task_stats_series = pd.Series(task_stats, index = output_csv_col)
        output_csv_df = output_csv_df.append(task_stats_series, ignore_index=True)
    
    output_csv_df.to_csv(output_csv_file_name)

    return



if __name__ == "__main__":

	main()






