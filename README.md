# Planaria: Spatially Multi-Tenant DNN Acceleration through Dynamic Architecture Fission

This codebase provides a simulation framework for **Planraia**, which is a spatially multi-tenant DNN accelerator. The codebase includes a hardware simulator for the planaria hardware that calculates the cycle counts and energy consumption of DNNs, while running on their best optimized fission configurations, in addition to Planraia's spatially multi-tenant scheduler that generates a workloads and shcdules it.

This document will help you to get the tool up and running.

To get more details about the Planaria hardware and its scheduling mechanisms, please refer to our MICRO'20 [Ghodrati20](https://www.microarch.org/micro53/papers/738300a681.pdf) paper.

If you find this tool useful in your research, we kindly request to cite our paper below:
* S. Ghodrati et.al., "Planaria: Dynamic Architecture Fission for Spatial Multi-Tenant Acceleration of Deep Neural Netwroks," MICRO, October 2020.

### Requirements
To use the simulator and scheduler you require to have Python >= 3.7 and following dependencies:
* Numpy
* Pandas
* Jupyter Notebook

### Execution Steps
This tool consists of two major steps:

#### Step (1): Hardware simulation and profiling:
In this step, you can simulate the execution of DNNs in isolation on Planaria hardware. The tool finds the optimal fission configuration of subarrays given a number of active subarrays for each DNN layer and generates two CSV files in `./numbers/` directory: (i) a more comprehensive performance reports that lists the optimal fission configurations, compute cycles, memory wait cycles, number of offchip/onchip memory accesses, loop tiling configurations, energy, and et.c. for each DNN layer (ii) a performance summary table, used during **scheduling** phase, that reports the optimal fission configuration, total cycles, energy, and the number of tiles for each DNN layer. The scheduler consults these summary tables to find the optimal spatial task co-location. To execute this step, consider following points:
* To run the profiling step, open the `experiments_notebook.ipynb` and follow the steps there.
* The hardware configuration parameters are defined in `cmx-{}.ini` files under the directory of `./configs/`. These configuration files are used to set the parameters of the accelerator in terms of memory bandwidth, on-chip buffer sizes, suabarray dimensions, number of active subarrays (the number of subarrays used for a DNN task), and etc. Please note that, to be able to use the scheduler, the profiling step for all possible number of active subarrays from 1 to 16 needs to happen. All the 16 config files are availabe under `./configs/` directory.
* The final performance and fission configurations are all generated in `./numbers/` directory and can be used during scheduling phase.
* The DNN graphs and their layer dimensions are implemented in `./src/benchmarks/benchmarks.py` and other DNNs can be implemented similarly.

#### Step (2): Spatial Task Scheduling:
In this step, a multi-tenant workload can be generated and be scheduled on Planaria hardware. To execute this step, consider following points:
* To run the scheduler, run the `run.py` file under `./scheduler` directory.
* A config file (`scheduler.ini`) is used to set the parameters of the scheduler. In this file, you can specify the frequnency of the accelerator, total number of active subarrays (default is 16), number of tasks in the multi-tenant workload, the arrival time distribution (options are poisson and uniform), the poisson parameter for arrival time, the type of DNN tasks in the workload scenario and their Quality of Service (QoS), and the path to the performance summary tables.
* After finishing the scheduling, a final CSV file is generated which reports the task statistcis including their arrival time, their end time, total execution time, QoS, and their ideal execution time if they were executed in isolation using all the subarrays.



* Note: We built our hardware simulator on top of [BitFusion](https://github.com/hsharma35/bitfusion) simulator, which simulates a single monolithic systolic array adn modified it to enable architecture fission and used the API from [nn_dataflow](https://github.com/stanford-mast/nn_dataflow) for DNN layer definitions.

