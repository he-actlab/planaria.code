from src.utils.utils import ceil_a_by_b, log2
from src.simulator.stats import Stats
import copy
import math

class CMX(object):
    def __init__(self, num_cores, num_pods, N, M, wgt_w, act_w, out_w, sram, frequency, mem_if_width, mem_channels, data_dispatch_width,
                                            wgt_bus_width, l2_act_sram_access_delay, l2_out_sram_access_delay, dram_cost, interconnect_cost):

        """
        Composable Chip Multi-Accelerator object
        """
        self.num_cores = num_cores
        self.num_pods = num_pods
        self.N = N
        self.M = M
        self.wgt_w = wgt_w
        self.act_w = act_w
        self.out_w = out_w
        self.sram = sram
        self.frequency = frequency
        self.mem_if_width = mem_if_width
        self.mem_channels = mem_channels
        self.data_dispatch_width = data_dispatch_width
        self.wgt_bus_width = wgt_bus_width
        self.l2_act_sram_access_delay = l2_act_sram_access_delay
        self.l2_out_sram_access_delay = l2_out_sram_access_delay
        self.dram_cost = dram_cost
        self.interconnect_cost = interconnect_cost

        ### FIFO and Accumulators
        self.fifo_capacity_ratio = 0.
        self.accumulator_capacity_ratio = 0.

        num_cores_per_pod, systolic_N, systolic_M, wgt_w, act_w, out_w, pod_sram, mem_channel_per_pod, pod_if_width,\
               wgt_bus_width, data_dispatch_width, l2_act_sram_access_delay,\
                l2_out_sram_access_delay, frequency = self.get_accelerator_pod_params()

        accelerator_pod_dict = {}
        accelerator_pod_id = ['AxPod ID: {}'.format(i) for i in range(int(self.num_pods))]
        for id in accelerator_pod_id:
            accelerator_pod_dict[id] = AcceleratorPod(num_cores_per_pod, systolic_N, systolic_M, wgt_w, act_w, out_w, pod_sram,\
                                                    mem_channel_per_pod, pod_if_width, wgt_bus_width, data_dispatch_width,\
                                                    l2_act_sram_access_delay, l2_out_sram_access_delay, frequency)
        self.accelerator_pod_dict = accelerator_pod_dict




    def get_accelerator_pod_params(self):
        """
        Generate the parameters of an Accelerator Pod
        """
        num_pods = self.num_pods
        num_cores = self.num_cores
        systolic_N = self.N
        systolic_M = self.M

        num_cores_per_pod = num_cores / num_pods

        pod_sram = {}
        wgt_sram_tot = self.sram['wgt']
        act_sram_tot = self.sram['act']
        wgt_sram_pod = wgt_sram_tot / num_pods
        act_sram_pod = act_sram_tot / num_pods
        pod_sram['wgt'] = wgt_sram_pod
        pod_sram['act'] = act_sram_pod

        mem_channel_per_pod = self.mem_channels / float(num_pods)
        pod_if_width = (mem_channel_per_pod / float(self.mem_channels)) * self.mem_if_width

        return num_cores_per_pod, systolic_N, systolic_M, self.wgt_w, self.act_w, self.out_w, pod_sram, mem_channel_per_pod, pod_if_width, self.wgt_bus_width, \
                                                             self.data_dispatch_width, self.l2_act_sram_access_delay, self.l2_out_sram_access_delay, self.frequency


class AcceleratorPod(object):
    def __init__(self, num_cores_per_pod, systolic_N, systolic_M, wgt_w, act_w, out_w, pod_sram, mem_channel_per_pod, pod_if_width,
                                                wgt_bus_width, data_dispatch_width, l2_act_sram_access_delay, l2_out_sram_access_delay, frequency):
        """
        AcceleratorPod Object
        """
        self.num_cores_in_pod = num_cores_per_pod
        self.systolic_N = systolic_N
        self.systolic_M = systolic_M
        self.wgt_w = wgt_w
        self.act_w = act_w
        self.out_w = out_w
        self.pod_sram = pod_sram
        self.l2_act_sram_access_delay = l2_act_sram_access_delay
        self.l2_out_sram_access_delay = l2_out_sram_access_delay
        self.frequency = frequency
        ### Busing system and off-chip memory interface
        self.mem_channel_in_pod = mem_channel_per_pod
        self.pod_if_width = pod_if_width
        self.wgt_bus_width = wgt_bus_width
        self.data_dispatch_width = data_dispatch_width


### Memory system
        ### SRAM Blocks
        self.core_sram = {}
        self.core_sram['wgt'] = self.pod_sram['wgt'] / self.num_cores_in_pod

        ### Considering seperate blocks for input and output activations.
        ### For now, I consider equal size on-chip memory for both input activations and output activations
        self.sram_block_in_pod = self.num_cores_in_pod * 2
        self.sram_block_act = self.num_cores_in_pod
        self.sram_block_out = self.num_cores_in_pod


        ### FIFO and Accumulators
        self.fifo_capacity_ratio = 0.
        self.accumulator_capacity_ratio = 0.

        self.core_act_fifo = self.pod_sram['act'] / 2 / self.num_cores_in_pod * self.fifo_capacity_ratio
        self.core_accumulator_fifo = self.pod_sram['act'] / 2 / self.num_cores_in_pod * self.fifo_capacity_ratio

        ### Bandwidth for loading the FIFOs;
        ### TODO: may need to be updated based on the word size!
        ### It is just set to a big number, since the actual design does not need FIFO/Accumulators
        self.load_fifo_width = self.systolic_N * self.act_w * 65536
        self.store_accumulator_width = self.systolic_M * self.out_w * 65536

        self.core_sram['act'] = self.pod_sram['act'] / 2 / self.num_cores_in_pod * (1 - self.fifo_capacity_ratio)
        self.core_sram['out'] = self.pod_sram['act'] / 2 / self.num_cores_in_pod * (1 - self.fifo_capacity_ratio)


        core_pod_dict = {}
        core_pod_id = ['Core ID: {}'.format(i) for i in range(int(self.num_cores_in_pod))]
        for id in core_pod_id:
            ### Considering 1 sram block for each core / 1 fifo/accum per core
            core_pod_dict[id] = Core(self.systolic_N, self.systolic_M, self.wgt_w, self.act_w, self.out_w, self.core_act_fifo,\
                                                    self.core_accumulator_fifo, 1,1 , self.core_sram['wgt'], self.core_sram['act'], self.core_sram['out'],\
                                                    1, 1, self.l2_act_sram_access_delay, self.l2_out_sram_access_delay, self.frequency)
        self.core_pod_dict = core_pod_dict

    def compose_core_across_pod(self, composed_core_config, composed_pod_config):
        """
        This fucntion creates an object of composed Ax pods and return that
        """

        composed_pod_core = copy.deepcopy(self.core_pod_dict['Core ID: 0'])
        composed_pod_core.set_compose_core_across_pod_param(composed_core_config, composed_pod_config)

        return composed_pod_core

    

    def compose_core_in_pod(self, config, num_available_cores):
        """
        This function creates a dictionary of composed cores in a pod and returns that
        """
        num_composed_cores = num_available_cores / (config[0] * config[1])
        composed_core_pod_dict = {}

        for i in range(int(num_composed_cores)):
            composed_core_pod_dict['Composed Core ID: {}'.format(i)] = copy.deepcopy(self.core_pod_dict['Core ID: {}'.format(i)])
            composed_core_pod_dict['Composed Core ID: {}'.format(i)].set_compose_core_in_pod_param(config)

        return composed_core_pod_dict


class Core(object):
    def __init__(self, N, M, wgt_w, act_w, out_w, act_fifo, accum_fifo, num_act_fifo, num_accum_fifo, wgt_sram, act_l2_sram, out_l2_sram, act_l2_sram_block, out_l2_sram_block,
                                                            l2_act_sram_access_delay, l2_out_sram_access_delay,
                                                            frequency):
        """
        Core Accelerator Object
        """
        self.N = N
        self.M = M
        self.wgt_w = wgt_w
        self.act_w = act_w
        self.out_w = out_w
        self.act_fifo = act_fifo
        self.accum_fifo = accum_fifo
        self.num_act_fifo = num_act_fifo
        self.num_accum_fifo = num_accum_fifo
        self.wgt_sram = wgt_sram
        self.act_l2_sram = act_l2_sram
        self.out_l2_sram = out_l2_sram
        self.act_l2_sram_block = act_l2_sram_block
        self.out_l2_sram_block = out_l2_sram_block
        self.l2_act_sram_access_delay = l2_act_sram_access_delay
        self.l2_out_sram_access_delay = l2_out_sram_access_delay

        self.load_fifo_width = self.N * self.act_w * 65536
        self.store_accumulator_width = self.M * self.out_w * 65536
        self.frequency = frequency

    def set_compose_core_in_pod_param(self, config):
        """
        This function compose accelerator cores in a pod.
        Compostion of cores at pod level leads to upgrading both compute and memory resources
        """
        compose_input_dim = config[0]
        compose_output_dim = config[1]

        self.N = self.N * compose_input_dim
        self.M = self.M * compose_output_dim
        self.act_fifo = self.act_fifo * compose_input_dim
        self.accum_fifo = self.accum_fifo * compose_output_dim
        self.wgt_sram = self.wgt_sram * compose_output_dim * compose_input_dim
        self.act_l2_sram_block = self.act_l2_sram_block * compose_input_dim * compose_output_dim
        self.out_l2_sram_block = self.out_l2_sram_block * compose_input_dim * compose_output_dim
        self.act_l2_sram = self.act_l2_sram * compose_output_dim * compose_input_dim
        self.out_l2_sram = self.out_l2_sram * compose_output_dim * compose_input_dim

        return

    def set_compose_core_across_pod_param(self, composed_core_config, composed_pod_config):
        """
        This function compose accelrator cores across multiple pods.
        Composition at this level increases the compute linearly but not memory
        """
        compose_pod_input_dim = composed_pod_config[0]
        compose_pod_output_dim = composed_pod_config[1]

        core_in_pod_composition_config = (composed_core_config[0]/composed_pod_config[0], composed_core_config[1]/ composed_pod_config[1])
        self.set_compose_core_in_pod_param(core_in_pod_composition_config)

        self.N = self.N * compose_pod_input_dim
        self.M = self.M * compose_pod_output_dim
        self.act_fifo = self.act_fifo * compose_pod_input_dim
        self.accum_fifo = self.accum_fifo * compose_pod_output_dim
        self.wgt_sram = self.wgt_sram * compose_pod_input_dim * compose_pod_output_dim

        if compose_pod_input_dim == compose_pod_output_dim:

            self.act_l2_sram_block = self.act_l2_sram_block * compose_pod_input_dim
            self.out_l2_sram_block = self.out_l2_sram_block * compose_pod_output_dim
            self.act_l2_sram = self.act_l2_sram * compose_pod_input_dim
            self.out_l2_sram = self.out_l2_sram * compose_pod_output_dim
        else:
            max_compose_dim = max(compose_pod_input_dim, compose_pod_output_dim)
            if max_compose_dim == compose_pod_input_dim:
                self.act_l2_sram_block = self.act_l2_sram_block * compose_pod_input_dim * compose_pod_output_dim
                self.out_l2_sram_block = self.out_l2_sram_block * compose_pod_output_dim
                self.act_l2_sram = self.act_l2_sram * compose_pod_input_dim * compose_pod_output_dim
                self.out_l2_sram = self.out_l2_sram * compose_pod_output_dim
            elif max_compose_dim == compose_pod_output_dim:
                self.act_l2_sram_block = self.act_l2_sram_block * compose_pod_input_dim
                self.out_l2_sram_block = self.out_l2_sram_block * compose_pod_output_dim * compose_pod_output_dim
                self.act_l2_sram = self.act_l2_sram * compose_pod_input_dim
                self.out_l2_sram = self.out_l2_sram * compose_pod_input_dim * compose_pod_output_dim

        return




    def get_core_compute_cycles(self, ic, oc, ow, oh, b, kw, kh, im2col=True):
        """
        Compute instruction
        args:
            ic: Input Channels
            oc: Output Channels
            ow: Output Width
            oh: Output Height
            kw: Output Height
            kh: Output Height
            b: Batch Size
            im2col: boolean. If true, we assume the cpu does im2col. Otherwise,
                    we do convolutions channel-wise
        """
        overhead = 0
        if im2col:
            ni = kw * kh * ic
            no = oc
            batch = b * oh * ow
            compute_cycles = batch * ceil_a_by_b(no, self.M) * \
                    (ceil_a_by_b(ni, self.N) + overhead)
        else:
            compute_cycles = b * ceil_a_by_b(oc, self.M) * \
                    ow * oh * kw * kh * \
                    (ceil_a_by_b(ic, self.N) + overhead)

        return compute_cycles


    def get_core_cycles(self, ic, oc, ow, oh, b, kw, kh,
                                    stats_reads_act, stats_writes_out):
        """
        Calculate the total number of cycels taken to compute a tile on a composed_core,
        considering compute + ld/str fifo/Accumulators
        """

        compute_cycles = self.get_core_compute_cycles(ic, oc, ow, oh, b, kw, kh, im2col=True)

        ### We do not consider any FIFO/Accumulator
        total_core_cycles = compute_cycles



        return total_core_cycles, compute_cycles
