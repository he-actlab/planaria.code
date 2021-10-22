import logging
import math
import configparser
import numpy as np

from nn_dataflow import ConvLayer
from nn_dataflow.Layer import DWConvLayer

from src.utils.utils import ceil_a_by_b, log2, lookup_pandas_dataframe, is_power_two, largest_power_of_two_divisor
from src.simulator.stats import Stats
from src.optimizer.optimizer import optimize_for_order, get_stats_fast
from src.simulator.accelerator import CMX, AcceleratorPod, Core

from sram.cacti_sweep import CactiSweep
import os
import pandas

class Simulator(object):
    """
    Simulator class
    """

    def __init__(self, config_file='conf.ini', verbose=False, energy_costs=None):

        # custom energy cost
        self.energy_costs = energy_costs

        self.config_file = config_file

        self.config = configparser.ConfigParser()
        self.config.read(config_file)



        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        # logging.basicConfig(level=log_level)
        self.logger = logging.getLogger('{}.{}'.format(__name__, 'Simulator'))
        self.logger.setLevel(log_level)
        self.logger.debug("Creating Simulator Object")


### System
        mem_if_width = self.config.getint('system', 'if_width')
        self.logger.debug("Memory Interface Bit-Width: {}-bits".format(mem_if_width))
        mem_channels = self.config.getint('system', 'mem_channels')
        self.logger.debug("Number of Off-chip Memory Channels: {}".format(mem_channels))
        data_dispatch_width = self.config.getint('system', 'data_dispatch_width')
        self.logger.debug("Data Dispatch On-chip Bitwidth: {}".format(data_dispatch_width))
        wgt_bus_width = self.config.getint('system', 'wgt_bus_width')
        self.logger.debug("Weight Bus Bitwidth: {}".format(wgt_bus_width))

### Accelrator
        systolic_dim = [self.config.getint('accelerator', 'row'),
                             1,
                             self.config.getint('accelerator', 'col')]
        self.logger.debug("Systolic Array dimentions for each core: {}".format(systolic_dim))
        num_cores = self.config.getint('accelerator', 'num_cores')
        self.logger.debug("Number of Accelerator Cores: {}".format(num_cores))
        num_pods = self.config.getint('accelerator', 'num_pods')
        self.logger.debug("Number of Accelerator Pods: {}".format(num_pods))
        num_active_cores = self.config.getint('accelerator', 'num_active_cores')
        self.logger.debug("Number of Ative Cores: {}".format(num_active_cores))
        self.num_active_cores = num_active_cores

        wgt_w = self.config.getint('accelerator', 'wgt_w')
        self.logger.debug("Weight Bitwidth: {}".format(wgt_w))
        l2_act_sram_access_delay = self.config.getint('accelerator', 'l2_act_sram_access_delay')
        self.logger.debug("Level2 Input Activation SRAM Access Delay: {}".format(l2_act_sram_access_delay))
        l2_out_sram_access_delay = self.config.getint('accelerator', 'l2_out_sram_access_delay')
        self.logger.debug("Level2 Output SRAM Access Delay: {}".format(l2_out_sram_access_delay))
        act_w = self.config.getint('accelerator', 'act_w')
        self.logger.debug("Input Activation Bitwidth: {}".format(wgt_w))
        out_w = self.config.getint('accelerator', 'out_w')
        self.logger.debug("Output Bitwidth: {}".format(out_w))

        # Using half the size assuming double buffering
        sram = {}

        sram['act'] = self.config.getint('accelerator', 'Act_SRAM')
        self.logger.debug("Total Activation SRAM size: {:,} Bytes".format(sram['act']))

        sram['wgt'] = self.config.getint('accelerator', 'Wgt_SRAM')
        self.logger.debug("Total Weight SRAM size: {:,} Bytes".format(sram['wgt']))

        frequency = self.config.getint('accelerator', 'frequency')
        self.logger.debug('Frequency: {:,} Hz'.format(frequency))

### DRAM
        dram_cost = self.config.getfloat('dram', 'dram_cost')
        self.logger.debug('DRAM Access Energy: {} pJ/bit'.format(dram_cost))
### Interconnect
        wgt_bus_cost = self.config.getfloat('noc', 'wgt_bus_cost')
        self.logger.debug('Wgt Bus Access Energy: {} pJ/bit'.format(wgt_bus_cost))
        data_dispatch_cost = self.config.getfloat('noc', 'data_dispatch_cost')
        self.logger.debug('On-chip Data Dispatch Access Energy: {} pJ/bit'.format(data_dispatch_cost))
        interconnect_cost = {'wgt_bus_cost': wgt_bus_cost, 'data_dispatch_cost': data_dispatch_cost}

        hp_peak_throughput = systolic_dim[0] * \
                             systolic_dim[1] * \
                             systolic_dim[2]

        N = systolic_dim[0]
        beta = systolic_dim[1]
        M = systolic_dim[2]
        assert beta == 1

        ##################################################
        # Get stats for SRAM
        tech_node = 45
        sram_csv = 'hardware_sweep/sram_results.csv'
        sram_opt_dict = {'technology (u)': tech_node*1.e-3}
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sram')
        self.sram_obj = CactiSweep(
                bin_file=os.path.join(dir_path, 'cacti/cacti'),
                csv_file=os.path.join(dir_path, 'cacti_sweep.csv'),
                default_json=os.path.join(dir_path, 'default.json'),
                default_dict=sram_opt_dict)


### Creating a DeComposable Multi Accelerator Object
        self.cmx = CMX(num_cores, num_pods, N, M, wgt_w, act_w, out_w, sram, frequency, mem_if_width, mem_channels,\
                                                                 data_dispatch_width, wgt_bus_width, l2_act_sram_access_delay,\
                                                                     l2_out_sram_access_delay, dram_cost, interconnect_cost)

        self.possible_composition_configs = self.get_possible_composition_configs(num_active_cores)




    def map_set_composed_core(self, num_threads, composed_core_configs):
        """
        This function maps the active cores to the accelerator pods and set the configuration of accelerators in each pod
        Since we are looking at homogenous composition, a configuration will be returned which is used for all composed cores
        """
        num_active_pods = math.ceil(self.num_active_cores / float(self.cmx.num_pods))
        executing_pod_core = {}

        if composed_core_configs[0] * composed_core_configs[1] > (self.cmx.num_cores / self.cmx.num_pods):
            ### Multiple pods need to be composed
            num_pods_to_be_composed = (composed_core_configs[0] * composed_core_configs[1]) / (self.cmx.num_cores / self.cmx.num_pods)


            ### Following is based on 16 subarrays and 4 pods
            configs_need_pods_to_compose_list = [(1,16), (16,1), (2,8), (8,2), (4,4), (1,8), (8,1), (2,4), (4,2)]
            pod_composition_configs_list = [(1,4), (4,1), (2,4), (4,2), (2,2), (1,2), (2,1), (1,2), (2,1)]

            pod_composition_config = pod_composition_configs_list[configs_need_pods_to_compose_list.index(composed_core_configs)]

            for i in range(int(num_threads)):
                executing_pod_core['Busy Composed Core AxPod ID: {}'.format(i)] = \
                                                        self.cmx.accelerator_pod_dict['AxPod ID: {}'.format(i)].compose_core_across_pod(composed_core_configs,
                                                                                                                                     pod_composition_config)

        else:
            ### no need to compose Pods
            num_complete_busy_pods = math.floor(self.num_active_cores / float(self.cmx.num_cores / self.cmx.num_pods))
            num_busy_pods = math.ceil(self.num_active_cores / float(self.cmx.num_cores / self.cmx.num_pods))
            num_composed_core_in_pod = (self.cmx.num_cores / self.cmx.num_pods) / (composed_core_configs[0] * composed_core_configs[1])


            if num_busy_pods == num_complete_busy_pods:
                ### All the cores in a pod are available
                for i in range(int(num_busy_pods)):
                        executing_pod_core['Busy AxPod ID: {}'.format(i)] = self.cmx.accelerator_pod_dict['AxPod ID: {}'.format(i)].compose_core_in_pod(composed_core_configs,\
                                                                                                                                     float(self.cmx.num_cores / self.cmx.num_pods))
            else:

                for i in range(int(num_busy_pods)):
                    if i in range(int(num_complete_busy_pods)):
                        executing_pod_core['Busy AxPod ID: {}'.format(i)] = self.cmx.accelerator_pod_dict['AxPod ID: {}'.format(i)].compose_core_in_pod(composed_core_configs,\
                                                                                                                float(self.cmx.num_cores / self.cmx.num_pods))
                    else:
                        num_availabe_cores = self.num_active_cores - num_complete_busy_pods * float(self.cmx.num_cores / self.cmx.num_pods)
                        executing_pod_core['Busy AxPod ID: {}'.format(i)] = self.cmx.accelerator_pod_dict['AxPod ID: {}'.format(i)].compose_core_in_pod(composed_core_configs,\
                                                                                                            num_availabe_cores)
            ### Assertion check to make sure that the number of threads match the number of composed cores
            assert num_threads == sum(len(executing_pod_core['Busy AxPod ID: {}'.format(i)]) for i in range(int(num_busy_pods)))

        return executing_pod_core




    def get_core_composition_configs(self, num_cores):
        ### get the homogenous configuration possibilities for a power of two core numbers
        one_d_possibilities = []
        d = 0
        while 2**d <= num_cores:
            one_d_possibilities.append(2**d)
            d += 1

        configs = []
        for i in one_d_possibilities:
            for j in one_d_possibilities:
                if i * j <= num_cores:
                    configs.append((i,j))
        return configs


    def get_possible_composition_configs(self, num_active_cores):
        """
        This function returns the possible composition configs based on the total number of cores, active cores and pods
        For now, we assume homogenous configs
        """
        num_cores_base = largest_power_of_two_divisor(num_active_cores)
        composed_core_configs_list = self.get_core_composition_configs(num_cores_base)
        num_threads_list = [num_active_cores/(c[0] * c[1]) for c in composed_core_configs_list]

        return num_threads_list, composed_core_configs_list


    def get_energy_cost(self):

        if self.energy_costs is not None:
            return self.energy_costs


        frequency = self.cmx.frequency
        ##################################################
        N = self.cmx.N
        M = self.cmx.M

        tot_wgt_sram_size = self.cmx.sram['wgt'] * 8
        tot_activation_size = self.cmx.sram['act'] * 8
        tot_act_size = tot_activation_size / 2
        tot_out_size = tot_activation_size / 2
        tot_act_fifo_size = tot_act_size * self.cmx.fifo_capacity_ratio
        tot_out_accum_size = tot_out_size * self.cmx.accumulator_capacity_ratio
        tot_act_sram_size = tot_act_size * (1 - self.cmx.fifo_capacity_ratio)
        tot_out_sram_size = tot_out_size * (1 - self.cmx.accumulator_capacity_ratio)

        tot_wgt_sram_bank = N * M * self.cmx.num_cores
        tot_act_fifo_bank = N * self.cmx.num_cores
        tot_out_accum_bank = M * self.cmx.num_cores
        ### Number of banks for act and out will be determined based on their capacity
        ### Number of blocks for act and out sram
        tot_act_sram_block = self.cmx.num_cores
        tot_out_sram_block = self.cmx.num_cores
        tot_act_sram_bank = tot_act_sram_block * N
        tot_out_sram_bank = tot_out_sram_block * M

        wgt_sram_bits = self.cmx.wgt_w
        act_sram_bits = self.cmx.act_w
        out_sram_bits = self.cmx.out_w
        act_fifo_bits = self.cmx.act_w
        out_accum_bits = self.cmx.out_w




        wgt_sram_word = ceil_a_by_b(tot_wgt_sram_size, tot_wgt_sram_bank * wgt_sram_bits)
        act_sram_word = ceil_a_by_b(tot_act_sram_size, tot_act_sram_bank * act_sram_bits)
        out_sram_word = ceil_a_by_b(tot_out_sram_size, tot_out_sram_bank * out_sram_bits)
        act_fifo_word = ceil_a_by_b(tot_act_fifo_size, tot_act_fifo_bank * act_fifo_bits)
        out_accum_word = ceil_a_by_b(tot_out_accum_size, tot_out_accum_bank * out_accum_bits)


        wgt_sram_bank_size = wgt_sram_word * wgt_sram_bits
        act_sram_bank_size = act_sram_word * act_sram_bits
        out_sram_bank_size = out_sram_word * out_sram_bits
        act_fifo_bank_size = act_fifo_word * act_fifo_bits
        out_accum_bank_size = out_accum_word * out_accum_bits

        assert wgt_sram_bank_size * tot_wgt_sram_bank == tot_wgt_sram_size
        assert act_sram_bank_size * tot_act_sram_bank == tot_act_sram_size
        assert out_sram_bank_size * tot_out_sram_bank == tot_out_sram_size
        assert act_fifo_bank_size * tot_act_fifo_bank == tot_act_fifo_size
        assert out_accum_bank_size * tot_out_accum_bank == tot_out_accum_size


        ##################################################
        cfg_dict = {'size (bytes)': wgt_sram_bank_size /8., 'block size (bytes)': wgt_sram_bits/8., 'read-write port': 0}
        wgt_sram_data = self.sram_obj.get_data_clean(cfg_dict)
        wgt_sram_read_energy = float(wgt_sram_data['read_energy_nJ']) / wgt_sram_bits
        wgt_sram_write_energy = float(wgt_sram_data['write_energy_nJ']) / wgt_sram_bits
        wgt_sram_leak_power = float(wgt_sram_data['leak_power_mW']) * tot_wgt_sram_bank
        wgt_sram_area = float(wgt_sram_data['area_mm^2']) * tot_wgt_sram_bank


        cfg_dict = {'size (bytes)': act_sram_bank_size /8., 'block size (bytes)': act_sram_bits/8., 'read-write port': 0}
        act_sram_data = self.sram_obj.get_data_clean(cfg_dict)
        act_sram_read_energy = float(act_sram_data['read_energy_nJ']) / act_sram_bits
        act_sram_write_energy = float(act_sram_data['write_energy_nJ']) / act_sram_bits
        act_sram_leak_power = float(act_sram_data['leak_power_mW']) * tot_act_sram_bank
        act_sram_area = float(act_sram_data['area_mm^2']) * tot_act_sram_bank

        cfg_dict = {'size (bytes)': out_sram_bank_size /8., 'block size (bytes)': out_sram_bits/8., 'read-write port': 1}
        out_sram_data = self.sram_obj.get_data_clean(cfg_dict)
        out_sram_read_energy = float(out_sram_data['read_energy_nJ']) / out_sram_bits
        out_sram_write_energy = float(out_sram_data['write_energy_nJ']) / out_sram_bits
        out_sram_leak_power = float(out_sram_data['leak_power_mW']) * tot_out_sram_bank
        out_sram_area = float(out_sram_data['area_mm^2']) * tot_out_sram_bank


        cfg_dict = {'size (bytes)': act_fifo_bank_size /8., 'block size (bytes)': act_fifo_bits/8., 'read-write port': 0}
        act_fifo_data = self.sram_obj.get_data_clean(cfg_dict)
        # act_fifo_read_energy = float(act_fifo_data['read_energy_nJ']) / act_fifo_bits
        act_fifo_read_energy = 0
        # act_fifo_write_energy = float(act_fifo_data['write_energy_nJ']) / act_fifo_bits
        act_fifo_write_energy = 0
        act_fifo_leak_power = float(act_fifo_data['leak_power_mW']) * tot_act_fifo_bank
        act_fifo_area = float(act_fifo_data['area_mm^2']) * tot_act_fifo_bank


        cfg_dict = {'size (bytes)': out_accum_bank_size /8., 'block size (bytes)': out_accum_bits/8., 'read-write port': 0}
        out_accum_data = self.sram_obj.get_data_clean(cfg_dict)
        # out_accum_read_energy = float(out_accum_data['read_energy_nJ']) / out_accum_bits
        out_accum_read_energy = 0
        # out_accum_write_energy = float(out_accum_data['write_energy_nJ']) / out_accum_bits
        out_accum_write_energy = 0
        out_accum_leak_power = float(out_accum_data['leak_power_mW']) * tot_out_accum_bank
        out_accum_area = float(out_accum_data['area_mm^2']) * tot_out_accum_bank


        core_csv = os.path.join('./results', 'systolic_array_synth.csv')
        core_synth_data = pandas.read_csv(core_csv)

        lookup_dict = {}
        lookup_dict['Wgt Precision (bits)'] = self.cmx.wgt_w
        lookup_dict['Act Precision (bits)'] = self.cmx.act_w
        lookup_dict['Out Precision (bits)'] = self.cmx.out_w
        lookup_dict['N'] = N
        lookup_dict['M'] = M

        core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)


        core_area = float(core_data['Area (mm^2)'])
        core_dyn_power = float(core_data['Dynamic Power (nW)'])
        core_dyn_energy = core_dyn_power / float(core_data['Frequency'])
        core_leak_power = float(core_data['Leakage Power (nW)'])
        core_leak_energy = core_leak_power / float(core_data['Frequency'])

        tot_core_area = core_area * self.cmx.num_cores
        tot_core_dyn_power = core_dyn_power * self.num_active_cores
        tot_core_dyn_energy = core_dyn_energy * self.num_active_cores
        tot_core_leak_power = core_leak_power * self.num_active_cores
        tot_core_leak_energy = core_leak_energy * self.num_active_cores

        total_leak_energy = 0

        return total_leak_energy, core_dyn_energy, wgt_sram_read_energy, wgt_sram_write_energy, act_sram_read_energy, act_sram_write_energy, out_sram_read_energy, out_sram_write_energy, act_fifo_read_energy, act_fifo_write_energy, out_accum_read_energy, out_accum_write_energy


    def __str__(self):
        ret = ''
        ret += 'Simulator object'
        ret += '\n'
        ret += '\tWgt precision: {}'.format(self.cmx.wgt_w)
        ret += '\n'
        ret += '\tAct precision: {}'.format(self.cmx.act_w)
        ret += '\n'
        ret += '\tOut precision: {}'.format(self.cmx.out_w)
        ret += '\n'
        ret += '\tNumber of cores: {}'.format(self.cmx.num_cores)
        ret += '\n'
        ret += '\tNumber of Accelerator Pods: {}'.format(self.cmx.num_pods)
        ret += '\n'
        ret += '\tSystolic array size: {} -inputs x {} -outputs'.format(
                self.cmx.N,
                self.cmx.M)

        ret += '\n'
        ret += '\tWgt Sram size: {:,} Bytes'.format(self.cmx.sram['wgt'])
        ret += '\n'
        ret += '\tAct sram: {:,} Bytes'.format((self.cmx.sram['act']/2) * (1-self.cmx.fifo_capacity_ratio))
        ret += '\n'
        ret += '\tOut sram size: {:,} Bytes'.format((self.cmx.sram['act']/2) * (1-self.cmx.fifo_capacity_ratio))
        ret += '\n'
        ret += '\tAct FIFO size: {:,} Bytes'.format((self.cmx.sram['act']/2) * (self.cmx.fifo_capacity_ratio))
        ret += '\n'
        ret += '\tOutput Accumulators size: {:,} Bytes'.format((self.cmx.sram['act']/2) * (self.cmx.fifo_capacity_ratio))
        ret += '\n'
        ret += 'Double buffering enabled. Sizes of SRAM are halved'
        return ret



    def get_FC_cycles(self, Ni, No,
                      iprec, wprec,
                      batch_size=1):
        """
        Get number of cycles required for Fully-Connected Layer.

        args:
            Ni: Input neurons
            No: Output neurons
            batch_size: Batch size for FC layer
            iprec: Precision for activations (bits)
            wprec: Precision for weights (bits)
            batch_size: Batch size for the layer

        description:
            This function calls the get_conv_cycles function
        """
        total_cycles = self.get_conv_cycles(1, 1, 1, Ni, No, iprec, wprec, batch_size)

        return total_cycles



    def verify_optimizer_results(self, tiling, ordering):
        #print('Verifying numbers')
        pass

    def get_conv_cycles(self, K, O, S, IC, OC, iprec, wprec, batch_size=1, im2col=True):
        """
        This functions does an exhaustive search for finding the optimal
        Tiling and Ordering parameters

        """
        composition_config_stats_list = []
        possible_num_threads, possible_composition_configs = self.possible_composition_configs
        for i in range(len(possible_composition_configs)):
            num_threads = possible_num_threads[i]
            composition_config = possible_composition_configs[i]
            print('Compostion Config : {} , Num thread: {}'.format(composition_config, num_threads))


        ### To perform the conv/fc layers, first we need to create the core configurations and then map the data to that
            executing_pod_core = self.map_set_composed_core(num_threads, composition_config)

            ### Check to see if we have composed_pod or not
            key_composed_core_ax_pod = 'Busy Composed Core AxPod ID: 0'
            key_non_composed_ax_pod = 'Busy AxPod ID: 0'
            key_composed_core = 'Composed Core ID: 0'

            if key_composed_core_ax_pod in executing_pod_core.keys():

                core_M = executing_pod_core['Busy Composed Core AxPod ID: 0'].M
                core_N = executing_pod_core['Busy Composed Core AxPod ID: 0'].N
            else:
                core_M = executing_pod_core['Busy AxPod ID: 0']['Composed Core ID: 0'].M
                core_N = executing_pod_core['Busy AxPod ID: 0']['Composed Core ID: 0'].N


            print('Core Dimension : {} * {}'.format(core_N, core_M))
            B = batch_size
            num_O_tiles = int(math.ceil(log2(O))) + 1
            num_IC_tiles = int(math.ceil(log2(IC))) + 1
            num_OC_tiles = int(math.ceil(log2(math.ceil(float(OC)/(core_M))))) + 1
            num_B_tiles = int(math.ceil(log2(batch_size))) + 1

            self.logger.debug('Number of O Tiles: {}'.format(num_O_tiles))
            self.logger.debug('Number of IC Tiles: {}'.format(num_IC_tiles))
            self.logger.debug('Number of OC Tiles: {}'.format(num_OC_tiles))
            self.logger.debug('Number of B Tiles: {}'.format(num_B_tiles))

            best_instructions_dict = {}
            conv_params = self.cmx, executing_pod_core, K, O, S, IC, OC, B, iprec, wprec, im2col, self.get_energy_cost()

            best_tiling, best_partitioning, best_order, best_stationary = optimize_for_order(conv_params)


            self.verify_optimizer_results(best_tiling, best_order)
            stats, stationary = get_stats_fast(conv_params, best_tiling, best_partitioning, best_order)
            energy = stats.energy
            cycles = stats.total_cycles
            assert stationary == best_stationary

            composition_config_stats_list.append([stats, best_tiling, best_order, best_partitioning, best_stationary, energy, cycles, energy*cycles])


            best_cycles = stats.total_cycles

            num_ops = O * O * K * K * IC * OC * B

        best_config_stats = min(composition_config_stats_list, key=lambda x: x[-1])
        best_config_idx = composition_config_stats_list.index(best_config_stats)
        best_num_threads = possible_num_threads[best_config_idx]
        best_core_composition_config = possible_composition_configs[best_config_idx]

        best_executing_pod_core = self.map_set_composed_core(best_num_threads, best_core_composition_config)

        return best_config_stats[0], best_config_stats[1:5], self.cmx, best_executing_pod_core



    def get_cycles(self, layer, batch_size=1):
        if isinstance(layer, ConvLayer):
            print(layer)
            if isinstance(layer, DWConvLayer):
                stats, dataflow, cmx, executing_pod_core = self.get_conv_cycles(layer.sfil, layer.hofm, layer.htrd, 1, 1, layer.iprec, layer.wprec, batch_size, im2col=True)


                stats.total_cycles *= layer.nifm
                stats.mem_stall_cycles *= layer.nifm
                stats.energy *= layer.nifm
                for n in stats.namespaces:
                    stats.reads[n] *= layer.nifm
                    stats.writes[n] *= layer.nifm
                stats.data_dispatch_hops *= layer.nifm
                stats.wgt_bus_hops *= layer.nifm


                ic_tuple = dataflow[0]['IC/ic']
                ic_tile_tuple = (ic_tuple[0] * layer.nifm, ic_tuple[1])

                dataflow[0]['IC/ic']= ic_tile_tuple


                return stats, dataflow, cmx, executing_pod_core
            else:
                return self.get_conv_cycles(layer.sfil,  # K
                                        layer.hofm,  # Oh == Ow
                                        layer.htrd,  # S
                                        layer.nifm,  # NI
                                        layer.nofm,  # NO
                                        layer.iprec,  # Activation Precision
                                        layer.wprec,  # Weight Precision
                                        batch_size, # Batch Size
                                        im2col=True)  # Batch Size
