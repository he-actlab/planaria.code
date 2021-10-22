import pandas
import os
import logging

from src.simulator.simulator import Simulator
from src.utils.utils import lookup_pandas_dataframe
import src.benchmarks.benchmarks as benchmarks
from nn_dataflow import ConvLayer

class SimulatorSweep(object):
    def __init__(self, csv_filename, config_file='conf.ini', verbose=False):
        self.sim_obj = Simulator(config_file, verbose=False)
        self.csv_filename = csv_filename
        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        self.logger = logging.getLogger('{}.{}'.format(__name__, 'Simulator'))
        self.logger.setLevel(log_level)

        self.columns = ['N', 'M',
                'Number of Threads',
                'Number of Active Cores',
                'DRAM Cost (pJ/bit)',
                'Weight Bus Cost (pJ/bit)',
                'Data Dispatch Bus Cost (pJ/bit)',
                'Activation Precision (bits)', 'Weight Precision (bits)',
                'Network', 'Layer',
                'Total Cycles', 'Core Cycles', 'Core Compute Cycles', 'Memory wait cycles', 'Energy', 'Data Dispatch Hops', 'Weight Bus Hops',
                'Weight SRAM Read', 'Weight SRAM Write',
                'Output SRAM Read', 'Output SRAM Write',
                'Activation SRAM Read', 'Actiation SRAM Write',
                'Activation FIFO Read', 'Activation FIFO Write',
                'Output Acccumulator Read', 'Output Accumulator Write',
                'DRAM Read', 'DRAM Write',
                'Tiling', 'Ordering', 'Partitioning', 'Stationary',
                'Bandwidth (bits/cycle)',
                'Weight SRAM Size (bits)', 'Output SRAM Size (bits)', 'Input SRAM Size (bits)', 'Act FIFO Size (bits)', 'Out Accumulator Size (Bits)',
                'Batch size']


        self.sweep_df = pandas.DataFrame(columns=self.columns)

    def sweep(self, sim_obj, batch_size):

        data_line = []

        list_bench = benchmarks.benchlist
        for b in list_bench:
            print('Benchmark: ', b)

            if b == 'GNMT':
                nn1, nn2 = benchmarks.get_bench_nn(b)
            else:
                nn = benchmarks.get_bench_nn(b)

            cmx = sim_obj.cmx
            self.logger.info('Simulating Benchmark: {}'.format(b))
            self.logger.info('DRAM Cost = {} pJ/bit'.format(cmx.dram_cost))
            self.logger.info('Interconenct Cost = {} pJ/bit'.format(cmx.interconnect_cost))
            self.logger.info('Number of Active Cores: {}'.format(sim_obj.num_active_cores))
            self.logger.info('Activation Precision (bits): {}'.format(cmx.act_w))
            self.logger.info('Output Precision (bits): {}'.format(cmx.out_w))
            self.logger.info('Batch size: {}'.format(batch_size))
            self.logger.info('Bandwidth (bits/cycle): {}'.format(cmx.mem_if_width))




            if b == 'GNMT':
                stats_en = benchmarks.get_bench_numbers(nn1, sim_obj, batch_size)
                stats_de = benchmarks.get_bench_numbers(nn2, sim_obj, batch_size)
                input_seq = 32
                output_seq = 32
                for layer in stats_en.keys():
                    stats_en[layer][0] *= input_seq
                for layer in stats_de.keys():
                    stats_de[layer][0] *= output_seq
                stats_en.update(stats_de)
                stats = stats_en
            else:

                stats = benchmarks.get_bench_numbers(nn, sim_obj, batch_size)
            for layer in stats:
                stats_list = stats[layer]
                stats_stats = stats_list[0]
                stats_dataflow = stats_list[1]
                stats_cmx = stats_list[2]
                stats_exec_core_pod = stats_list[3]

                ### Ge the info of the composed Core/Pod
                key_composed_core_ax_pod = 'Busy Composed Core AxPod ID: 0'
                key_non_composed_ax_pod = 'Busy AxPod ID: 0'
                key_composed_core = 'Composed Core ID: 0'

                if key_composed_core_ax_pod in stats_exec_core_pod.keys():
                    ### We have composed Ax Pod

                    acc_core_obj = stats_exec_core_pod['Busy Composed Core AxPod ID: 0']
                    num_threads = len(stats_exec_core_pod.keys())
                else:
                    ### We just have composed Core
                    acc_core_obj = stats_exec_core_pod['Busy AxPod ID: 0']['Composed Core ID: 0']
                    num_busy_pods = len(stats_exec_core_pod.keys())
                    num_threads = sum(len(stats_exec_core_pod['Busy AxPod ID: {}'.format(i)]) for i in range(num_busy_pods))

                self.logger.info('Composed Core Dimension: {} * {}'.format(acc_core_obj.N, acc_core_obj.M))
                self.logger.info('Number of Threads: {}'.format(num_threads))

                data_line.append((acc_core_obj.N, acc_core_obj.M, num_threads, sim_obj.num_active_cores, cmx.dram_cost, cmx.interconnect_cost['wgt_bus_cost'],
                    cmx.interconnect_cost['data_dispatch_cost'], cmx.act_w, cmx.wgt_w, b, layer,
                    stats_stats.total_cycles, stats_stats.core_cycles, stats_stats.core_compute_cycles, stats_stats.mem_stall_cycles, stats_stats.energy, stats_stats.data_dispatch_hops,
                    stats_stats.wgt_bus_hops,
                    stats_stats.reads['wgt'], stats_stats.writes['wgt'],
                    stats_stats.reads['out'], stats_stats.writes['out'],
                    stats_stats.reads['act'], stats_stats.writes['act'],
                    stats_stats.reads['act_fifo'], stats_stats.writes['act_fifo'],
                    stats_stats.reads['out_accum'], stats_stats.writes['out_accum'],
                    stats_stats.reads['dram'], stats_stats.writes['dram'],
                    stats_dataflow[0], stats_dataflow[1], stats_dataflow[2], stats_dataflow[3],
                    cmx.mem_if_width,
                    acc_core_obj.wgt_sram, acc_core_obj.out_l2_sram, acc_core_obj.act_l2_sram, acc_core_obj.act_fifo, acc_core_obj.accum_fifo, batch_size))


        if len(data_line) > 0:

            self.sweep_df = self.sweep_df.append(pandas.DataFrame(data_line, columns=self.columns))

            data_line = []
        return self.sweep_df

def check_pandas_or_run(sim, sim_sweep_csv, batch_size, config_file):

    print('running new configuration...')
    sweep_obj = SimulatorSweep(sim_sweep_csv, config_file)
    dataframe = sweep_obj.sweep(sim, batch_size)

    ### Dumping the stats for NNs
    dataframe.to_csv(sim_sweep_csv, index=False)
    return dataframe
