import math
import functools
import time
import logging

from itertools import permutations
from multiprocessing import Pool, cpu_count

from src.utils.utils import ceil_a_by_b, log2, get_two_largest_divisor
from src.simulator.stats import Stats

import numpy as np

logger = logging.getLogger('{}.{}'.format(__name__, 'Optimizer'))
logger.setLevel(logging.DEBUG)

tile_deps = {}
tile_deps['B/b']   = {'act': True, 'wgt': False, 'out': True}
tile_deps['OW/ow'] = {'act': True, 'wgt': False, 'out': True}
tile_deps['OH/oh'] = {'act': True, 'wgt': False, 'out': True}
tile_deps['IC/ic'] = {'act': True, 'wgt': True, 'out': False}
tile_deps['OC/oc'] = {'act': False, 'wgt': True, 'out': True}


def get_stats_fast(conv_params, tiling, partitioning, order_type, verbose=False):
    """
    Returns cycles and memory accesses to DRAM, Activation SRAM, Weight SRAM, Out SRAM, Act FIFO, Out Accumulator
    """
    cmx, acc_cores_dict, K, O, S, IC, OC, B, iprec, wprec, im2col, energy_cost = conv_params

    im2col = True

    ### Note: The tile that we receive here is actually a sub_tile which is a partition of the main tile (it has been partitioned based on the number of threads)

    ### We first calculate the stats of a core and then get the stats for the whole accelerator

    num_b, b = tiling['B/b']
    num_ow, ow = tiling['OW/ow']
    num_oh, oh = tiling['OH/oh']
    num_ic, ic = tiling['IC/ic']
    num_oc, oc = tiling['OC/oc']

    kw = kh = K


    writes = {}
    reads = {}

    if im2col:
        writes['wgt'] = \
                K * K * ic  * \
                oc * \
                wprec 
    else:
        writes['wgt'] = \
                K * K * ic * \
                oc * \
                wprec
    if im2col:
        iw = K + (ow - 1) * S
        ih = K + (oh - 1) * S
        writes['act'] = iw * ih * ic * b * iprec
    else:
        iw = K + (ow - 1) * S
        ih = K + (oh - 1) * S
        writes['act'] = iw * ih * ic * b * iprec

    oprec = cmx.out_w
    writes['out'] = ow * oh * oc * b * oprec
    reads['out'] = ow * oh * oc * b * oprec

    ### Getting one accelerator core object
    key_composed_core_ax_pod = 'Busy Composed Core AxPod ID: 0'
    key_non_composed_ax_pod = 'Busy AxPod ID: 0'
    key_composed_core = 'Composed Core ID: 0'

    if key_composed_core_ax_pod in acc_cores_dict.keys():
        ### We have composed Ax Pod

        acc_core_obj = acc_cores_dict['Busy Composed Core AxPod ID: 0']
        num_threads = len(acc_cores_dict.keys())
    else:
        ### We just have composed Core
        acc_core_obj = acc_cores_dict['Busy AxPod ID: 0']['Composed Core ID: 0']
        num_busy_pods = len(acc_cores_dict.keys())
        num_threads = sum(len(acc_cores_dict['Busy AxPod ID: {}'.format(i)]) for i in range(int(num_busy_pods)))

    ### Accelerator core sram info
    core_sram = {}
    core_sram['wgt'] = acc_core_obj.wgt_sram
    core_sram['act'] = acc_core_obj.act_l2_sram
    core_sram['out'] = acc_core_obj.out_l2_sram

    overflow = False
    if writes['wgt'] > acc_core_obj.wgt_sram * 8 / 2:
        if verbose:
            print('wgt overflow: {}'.format(writes['wgt']))
            print(b, ow, oh, ic, oc)
        overflow = True
    if writes['act'] > acc_core_obj.act_l2_sram * 8 / 2:
        if verbose:
            print('act overflow')
            print(b, ow, oh, ic, oc)
        overflow = True
    if writes['out'] > acc_core_obj.out_l2_sram * 8 / 2:
        if verbose:
            print('out overflow')
            print(b, ow, oh, ic, oc)
        overflow = True
    if overflow:
        if verbose:
            print('Activation size: {} bytes'.format(writes['act']/8.))
            print('Weights size: {} bytes'.format(writes['wgt']/8.))
            print('Output size: {} bytes'.format(writes['out']/8.))
        return


    max_write_size = {}
    max_read_size = {}
    for namespace in writes:
        max_write_size[namespace] = writes[namespace]
    for namespace in reads:
        max_read_size[namespace] = reads[namespace]

    # First the loop block optimizations
    stats = Stats()
    write_promote = {'wgt': True, 'act': True, 'out': True}
    read_promote = {'out': True}
    if verbose:
        logger.debug('Initialize reads/writes')
        logger.debug('\tim2col: {}'.format(im2col))
        logger.debug('\tTiling: {}'.format(tiling))
        logger.debug('\tReads : {}'.format(reads))
        logger.debug('\tWrites: {}'.format(writes))
    for loop in reversed(order_type):
        num_tiles, tile_size = tiling[loop]
        # promote all writes
        for namespace in writes:
            # promote is true
            if write_promote[namespace]:
                # If tile loop depends on the namespace index, make the read size larger
                if tile_deps[loop][namespace]:
                    writes[namespace] *= num_tiles
                    # If tile size is larger than the SRAM, set promote to False
                    if writes[namespace] > core_sram[namespace]*8./2:
                        write_promote[namespace] = False
                    else:
                        max_write_size[namespace] = writes[namespace]
            else:
                writes[namespace] *= num_tiles

        # promote all reads
        for namespace in reads:
            # promote is true
            if read_promote[namespace]:
                # Tile loop depends on the namespace index
                if tile_deps[loop][namespace]:
                    reads[namespace] *= num_tiles
                    # Tile size is now larger than the SRAM, set promote to False
                    if reads[namespace] > core_sram[namespace]*8./2:
                        read_promote[namespace] = False
                    else:
                        max_read_size[namespace] = writes[namespace]
            else:
                reads[namespace] *= num_tiles

        if verbose:
            logger.debug('Loop: {}'.format(loop))
            logger.debug('\tLoop range: {}'.format(tiling[loop]))
            logger.debug('\tMax write size: {}'.format(max_write_size))
            logger.debug('\tMax read size: {}'.format(max_read_size))
            logger.debug('\tLoop Dependencies: {}'.format(tile_deps[loop]))
            logger.debug('\tLoop Promote: {}'.format(write_promote))
            logger.debug('\tReads : {}'.format(reads))
            logger.debug('\tWrites: {}'.format(writes))


    for namespace in writes:
        stats.writes[namespace] = writes[namespace]
        stats.reads['dram'] += writes[namespace]
    for namespace in reads:
        stats.reads[namespace] = reads[namespace]
        stats.writes['dram'] += reads[namespace]



    num_tiles = num_b * num_ow * num_oh * num_ic * num_oc

    ### We consider weight-stationary dataflow similar to TPU
    if verbose:
        logger.debug('SRAM access order: Weight Stationary')
    stationary = 'Weight Stationary'
    stats.reads['act'] += num_tiles * (kw * kh * ic) * (b * ow * oh) * iprec * math.ceil(float(oc)/(acc_core_obj.M))
    stats.reads['out'] += num_tiles * (oc) * (b * ow * oh) * oprec * math.ceil(float(kw * kh * ic)/float(acc_core_obj.N))
    stats.writes['out'] += num_tiles * (oc) * (b * ow * oh) * oprec * math.ceil(float(kw * kh * ic)/float(acc_core_obj.N))
    stats.reads['wgt'] += num_tiles * (kw * kh * ic * oc) * wprec
    ### Since we have FIFO/Accumulators, we rewrite the data from srams to fifo/accum and read it back from there.
    # As such, the number of read/write for fifo/accum is the same as srams
    ## Althoug we do not need FIFO/Accumulators but we consider its modularity in the simulator
    stats.reads['act_fifo'] = stats.reads['act']
    stats.writes['act_fifo'] = stats.writes['act']
    stats.reads['out_accum'] = stats.reads['out']
    stats.writes['out_accum'] = stats.writes['out']


    initial_dram_reads = 0
    final_dram_writes = 0

    for namespace in max_write_size:
        initial_dram_reads += max_write_size[namespace]

    if partitioning == 'oc':
        initial_dram_reads += (num_threads - 1) * max_write_size['wgt'] + (num_threads - 1) * max_write_size['out']
    else:
        initial_dram_reads += (num_threads - 1) * max_write_size['act'] + (num_threads - 1) * max_write_size['out']


    for namespace in max_read_size:
        final_dram_writes += max_read_size[namespace]

    final_dram_writes += (num_threads - 1) * max_read_size['out']


    if stats is not None:
    ### Getting info about busy pods and core
        num_busy_cores = num_threads * (acc_core_obj.N * acc_core_obj.M) / (cmx.N * cmx.M)
        mem_if_available_width = cmx.mem_if_width * (float(num_busy_cores) / float(cmx.num_cores))

        latency = math.ceil(initial_dram_reads / float(mem_if_available_width)) + \
                math.ceil(final_dram_writes / float(mem_if_available_width))


        if partitioning == 'oc':
            total_dram_read = stats.writes['act'] + stats.writes['wgt'] * num_threads + stats.writes['dram'] * num_threads
            total_dram_writes = stats.writes['dram'] * num_threads
        else:
            total_dram_read = stats.writes['act'] * num_threads + stats.writes['wgt'] + stats.writes['dram'] * num_threads
            total_dram_writes = stats.writes['dram'] * num_threads

        total_dram_accesses = total_dram_read + total_dram_writes

        middle_dram_accesses = total_dram_accesses - initial_dram_reads - final_dram_writes


        stats_reads_act_tile = (kw * kh * ic) * (b * ow * oh) * iprec * math.ceil(float(oc)/(acc_core_obj.M))
        stats_writes_out_tile = (oc) * (b * ow * oh) * oprec * math.ceil(float(kw * kh * ic)/float(acc_core_obj.N))

        core_cycles_tile, core_compute_cycles_tile = acc_core_obj.get_core_cycles(ic, oc, ow, oh, b, kw, kh,
                                                                    stats_reads_act_tile, stats_writes_out_tile)

        core_cycles = num_tiles * core_cycles_tile
        core_compute_cycles = num_tiles * core_compute_cycles_tile

        stats.core_cycles = core_cycles
        stats.core_compute_cycles = core_compute_cycles


        memory_cycles_required = ceil_a_by_b(middle_dram_accesses, mem_if_available_width)

        memory_stalls = max(0, memory_cycles_required - core_cycles) + latency
        stats.total_cycles = core_cycles + memory_stalls
        stats.mem_stall_cycles = memory_stalls

        stats.wgt_bus_hops = 0
        stats.data_dispatch_hops = 0

        stats.reads['act'] = stats.reads['act'] * num_threads
        stats.reads['wgt'] = stats.reads['wgt'] * num_threads
        stats.reads['out'] = stats.reads['out'] * num_threads
        stats.reads['act_fifo'] *= num_threads
        stats.reads['out_accum'] *= num_threads


        if partitioning == 'oc':


            stats.reads['dram'] = stats.writes['act'] + stats.writes['wgt'] * num_threads + stats.writes['dram'] * num_threads
            if key_composed_core_ax_pod in acc_cores_dict.keys():
                num_busy_pod_in_thread = ((acc_core_obj.N * acc_core_obj.M) / (cmx.N * cmx.M)) / (cmx.num_cores / cmx.num_pods)
                wgt_packet_size = stats.writes['wgt'] / (((acc_core_obj.N * acc_core_obj.M) / (cmx.N * cmx.M)))
                wgt_bus_hops_in_pod = wgt_packet_size * sum(range(int(cmx.num_cores / cmx.num_pods)))
                wgt_bus_hops_in_thread = wgt_bus_hops_in_pod * num_busy_pod_in_thread
                wgt_bus_hops = wgt_bus_hops_in_thread * num_threads
                stats.wgt_bus_hops = wgt_bus_hops

                stats.data_dispatch_hops = stats.writes['act'] * (num_threads - 1)


            else:
                ### check if all the cores in the pods are busy
                num_busy_pods = len(acc_cores_dict)
                if num_busy_pods * (cmx.num_cores / cmx.num_pods) == num_threads * (acc_core_obj.M * acc_core_obj.N) / (cmx.M * cmx.N):
                    ### All the cores in busy pods are busy and we don't have composed Ax pod
                    # we can caulculate for one pod and then scale with the number of pods
                    num_threads_in_pod = num_threads / num_busy_pods
                    wgt_packet_size_in_pod =  stats.writes['wgt'] / ((cmx.num_cores / cmx.num_pods) / num_threads_in_pod)
                    wgt_bus_hops_in_pod = wgt_packet_size_in_pod * sum(range(int(cmx.num_cores / cmx.num_pods)))
                    wgt_bus_hops = wgt_bus_hops_in_pod * num_busy_pods
                    stats.wgt_bus_hops = wgt_bus_hops

                    stats.data_dispatch_hops = stats.writes['act'] * (num_busy_pods - 1)
                else:
                    ### there is a pod which is not completely busy
                    num_complete_busy_pods = num_busy_pods - 1
                    num_threads_in_complete_busy_pod = (cmx.num_cores / cmx.num_pods) / ((acc_core_obj.M * acc_core_obj.N) / (cmx.M * cmx.N))
                    wgt_packet_size_in_complete_busy_pod =  stats.writes['wgt'] / ((cmx.num_cores / cmx.num_pods) / num_threads_in_complete_busy_pod)
                    wgt_bus_hops_in_complete_busy_pod = wgt_packet_size_in_complete_busy_pod * sum(range(int(cmx.num_cores / cmx.num_pods)))
                    wgt_bus_hops = wgt_bus_hops_in_complete_busy_pod * num_complete_busy_pods
                    ### Now we need to calculate the stats for the non_complete_busy pod
                    num_cores_in_partial_busy_pod = num_threads * ((acc_core_obj.M * acc_core_obj.N) / (cmx.M * cmx.N)) - \
                                                                num_complete_busy_pods * ((cmx.num_cores / cmx.num_pods))
                    num_threads_in_partial_busy_pod = num_threads - num_threads_in_complete_busy_pod * num_complete_busy_pods
                    ### thread size is equivalent in all the pods
                    wgt_packet_size_in_partial_busy_pod = wgt_packet_size_in_complete_busy_pod
                    wgt_bus_hops_in_partail_busy_pod = wgt_packet_size_in_partial_busy_pod * sum(range(int(num_cores_in_partial_busy_pod)))
                    wgt_bus_hops += wgt_bus_hops_in_partail_busy_pod
                    stats.wgt_bus_hops = wgt_bus_hops

                    stats.data_dispatch_hops = stats.writes['act'] * (num_busy_pods - 1)

        else:

            stats.reads['dram'] = stats.writes['wgt'] + stats.writes['act'] * num_threads + stats.writes['dram'] * num_threads
            if key_composed_core_ax_pod in acc_cores_dict.keys():
                num_busy_pod_in_thread = ((acc_core_obj.N * acc_core_obj.M) / (cmx.N * cmx.M)) / (cmx.num_cores / cmx.num_pods)
                wgt_packet_size = stats.writes['wgt'] / (((acc_core_obj.N * acc_core_obj.M) / (cmx.N * cmx.M)))
                wgt_bus_hops_in_pod = wgt_packet_size * sum(range(int(cmx.num_cores / cmx.num_pods)))
                wgt_bus_hops_in_thread = wgt_bus_hops_in_pod * num_busy_pod_in_thread
                stats.wgt_bus_hops = wgt_bus_hops_in_thread * num_threads
                stats.data_dispatch_hops = stats.writes['wgt'] * (num_threads - 1)


            else:
                ### check if all the cores in the pods are busy
                num_busy_pods = len(acc_cores_dict)
                if num_busy_pods * (cmx.num_cores / cmx.num_pods) == num_threads * (acc_core_obj.M * acc_core_obj.N) / (cmx.M * cmx.N):
                    ### All the cores in busy pods are busy and we don't have composed Ax pod
                    # we can caulculate for one pod and then scale with the number of pods
                    num_threads_in_pod = num_threads / num_busy_pods
                    wgt_packet_size_in_pod =  stats.writes['wgt'] / ((cmx.num_cores / cmx.num_pods) / num_threads_in_pod)

                    ### The below part is based on 4 cores (subarrays) in a Pod
                    if num_threads_in_pod == 1:
                        wgt_bus_hops_in_pod = wgt_packet_size_in_pod * sum(range(int(cmx.num_cores / cmx.num_pods)))

                    elif num_threads_in_pod == 2 and acc_core_obj.N < acc_core_obj.M:
                        wgt_bus_hops_in_pod = wgt_packet_size_in_pod * (3 + 2)

                    elif num_threads_in_pod == 2 and acc_core_obj.N > acc_core_obj.M:
                        wgt_bus_hops_in_pod = wgt_packet_size_in_pod * (2 + 2)

                    else:
                        ### num_threads == 4
                        wgt_bus_hops_in_pod = wgt_packet_size_in_pod * ((cmx.num_cores / cmx.num_pods) - 1)

                    wgt_bus_hops_across_pods = num_busy_pods * wgt_bus_hops_in_pod
                    stats.data_dispatch_hops = (num_busy_pods - 1) * stats.writes['wgt']
                    ### Since each pod has its own private activations and can read it from its own memory channel, there is no need to use the data_dispatch_interconnect.
                    stats.data_dispatch_hops += 0


                else:

                    num_complete_busy_pods = num_busy_pods - 1
                    num_threads_in_complete_busy_pod = (cmx.num_cores / cmx.num_pods) / ((acc_core_obj.M * acc_core_obj.N) / (cmx.M * cmx.N))
                    wgt_packet_size_in_complete_busy_pod =  stats.writes['wgt'] / ((cmx.num_cores / cmx.num_pods) / num_threads_in_complete_busy_pod)

                    if num_threads_in_complete_busy_pod == 2 and acc_core_obj.N < acc_core_obj.M:
                        wgt_bus_hops_in_complete_busy_pod = wgt_packet_size_in_complete_busy_pod * (3 + 2)
                        wgt_bus_hops_across_pods = wgt_bus_hops_in_complete_busy_pod * num_complete_busy_pods
                        wgt_bus_hops_across_pods += wgt_packet_size_in_complete_busy_pod

                        stats.wgt_bus_hops = wgt_bus_hops_across_pods
                        stats.data_dispatch_hops = (num_busy_pods - 1) * stats.writes['wgt']

                    elif num_threads_in_complete_busy_pod == 2 and acc_core_obj.N > acc_core_obj.M:
                        wgt_bus_hops_in_complete_busy_pod = wgt_packet_size_in_complete_busy_pod * (2 + 2)
                        wgt_bus_hops_across_pods = wgt_bus_hops_in_complete_busy_pod * num_complete_busy_pods
                        wgt_bus_hops_across_pods += 0

                        stats.wgt_bus_hops = wgt_bus_hops_across_pods
                        stats.data_dispatch_hops = (num_busy_pods - 1) * stats.writes['wgt']

                    else:
                        ### num_threads == 4
                        wgt_bus_hops_in_complete_busy_pod = wgt_packet_size_in_complete_busy_pod * ((cmx.num_cores / cmx.num_pods) - 1)
                        wgt_bus_hops_across_pods = wgt_bus_hops_in_complete_busy_pod * num_complete_busy_pods
                        wgt_bus_hops_across_pods += wgt_packet_size_in_complete_busy_pod * ((num_threads * (acc_core_obj.M * acc_core_obj.N) / (cmx.M * cmx.N)) - \
                                                                                                                       num_complete_busy_pods * (cmx.num_cores / cmx.num_pods) - 1)
                        stats.wgt_bus_hops = wgt_bus_hops_across_pods
                        stats.data_dispatch_hops = (num_busy_pods - 1) * stats.writes['wgt']



        stats.writes['act'] = stats.writes['act'] * num_threads
        stats.writes['wgt'] = stats.writes['wgt'] * num_threads
        stats.writes['out'] = stats.writes['out'] * num_threads
        stats.writes['act_fifo'] = stats.writes['act_fifo'] * num_threads
        stats.writes['out_accum'] = stats.writes['out_accum'] * num_threads
        stats.writes['dram'] = stats.writes['dram'] * num_threads

        dram_cost = cmx.dram_cost
        interconnect_cost = cmx.interconnect_cost
        stats.energy = stats.get_energy(energy_cost, dram_cost, interconnect_cost)



    return (stats, stationary)



def optimize_for_order(conv_params):
    # Generate permutations for the order
    loops = ['B/b', 'OW/ow', 'OH/oh', 'IC/ic', 'OC/oc']
    order = set(permutations(loops))

    return_dict = {}
    cmx, acc_cores_dict, K, O, S, IC, OC, B, iprec, wprec, im2col, energy_cost = conv_params

    _bound_optimizer_method = functools.partial(_optimize_for_order, conv_params)

    try:
        pool = Pool(cpu_count())
        results = pool.map_async(_bound_optimizer_method, order).get(10000)
        pool.close()
        pool.join()


        best_cycles = None
        best_energy = None
        min_cycles = min([x[-4] for x in results])
        min_energy = min([x[-3] for x in results])
        cycles_list = [x[-2] for x in results]
        energy_list = [x[-1] for x in results]
        for r in results:
            tiling, order_type, cycles, energy, partitioning, stationary = r
            if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy > energy):

                best_cycles = cycles
                best_energy = energy
                best_tiling = tiling
                best_order = order_type
                best_partitioning = partitioning
                best_stationary = stationary

        return best_tiling, best_partitioning, best_order, best_stationary

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        return



def _optimize_for_order(conv_params, order_type, verbose=False):
    """
    For a given ordering, optimizes tiling
    Args:
        conv_params: A tuple with convolution params
        order_type: ordering loop
    """
    cmx, acc_cores_dict, K, O, S, IC, OC, B, iprec, wprec, im2col, energy_cost = conv_params
    I = (O - 1) * S + K

    im2col = True
    # We do not tile the "K" dimension and compute an entire 2-D conv at a
    # time
    num_O_tiles = int(math.ceil(log2(O))) + 1
    num_OW_tiles = num_O_tiles
    num_OH_tiles = num_O_tiles

    num_IC_tiles = int(math.ceil(log2(IC))) + 1


    ### Check to see if we have composed_pod or not
    key_composed_core_ax_pod = 'Busy Composed Core AxPod ID: 0'
    key_non_composed_ax_pod = 'Busy AxPod ID: 0'
    key_composed_core = 'Composed Core ID: 0'

    if key_composed_core_ax_pod in acc_cores_dict.keys():

        core_M = acc_cores_dict['Busy Composed Core AxPod ID: 0'].M

        num_threads = len(acc_cores_dict.keys())
    else:
        core_M = acc_cores_dict['Busy AxPod ID: 0']['Composed Core ID: 0'].M

        num_busy_pods = len(acc_cores_dict.keys())
        num_threads = sum(len(acc_cores_dict['Busy AxPod ID: {}'.format(i)]) for i in range(int(num_busy_pods)))

    if im2col:
        num_OC_tiles = int(math.ceil(log2(OC))) + 1
    else:
        num_OC_tiles = int(math.ceil(log2(math.ceil(float(OC)/core_M)))) + 1

    num_B_tiles = int(math.ceil(log2(B))) + 1

    best_cycles = None
    best_energy = None
    best_tiling = None

    ### For now, we just look at batch_size = 1. So we just partition data across cores on OC and OF (OW, OH)
    partitioning_options = ['oc', 'of']


    for _b in range(int(num_B_tiles)):
        b = min(1 << _b, B)
        num_b = ceil_a_by_b(B, b)

        for _ow in range(int(num_OW_tiles)):
            ow = min(1 << _ow, O)
            num_ow = ceil_a_by_b(O, ow)

            for _oh in range(int(num_OH_tiles)):
                oh = min(1 << _oh, O)
                num_oh = ceil_a_by_b(O, oh)

                for _ic in range(int(num_IC_tiles)):
                    ic = min(1 << _ic, IC)
                    num_ic = ceil_a_by_b(IC, ic)

                    for _oc in range(int(num_OC_tiles)):

                        if im2col:
                            oc = min((1 << _oc), OC)
                        else:
                            oc = min((1 << _oc) * core_M * 8/wprec, OC)

                        num_oc = ceil_a_by_b(OC, oc)

                        iw = K + (ow - 1) * S
                        ih = K + (oh - 1) * S

                        for part in partitioning_options:

                            if part == 'oc':

                                oc_part_thread = math.ceil(float(oc)/float(num_threads))
                                ow_part_thread = math.ceil(float(ow)/float(1))
                                oh_part_thread = math.ceil(float(oh)/float(1))
                                tiling = {}
                                tiling['B/b'] = (num_b, b)
                                tiling['OW/ow'] = (num_ow, ow_part_thread)
                                tiling['OH/oh'] = (num_oh, oh_part_thread)
                                tiling['IC/ic'] = (num_ic, ic)
                                tiling['OC/oc'] = (num_oc, oc_part_thread)


                            else:

                                div_ow, div_oh = get_two_largest_divisor(num_threads)

                                ow_part_thread = math.ceil(float(ow)/float(div_ow))
                                oh_part_thread = math.ceil(float(oh)/float(div_oh))
                                oc_part_thread = math.ceil(float(oc)/float(1))
                                tiling = {}
                                tiling['B/b'] = (num_b, b)
                                tiling['OW/ow'] = (num_ow, ow_part_thread)
                                tiling['OH/oh'] = (num_oh, oh_part_thread)
                                tiling['IC/ic'] = (num_ic, ic)
                                tiling['OC/oc'] = (num_oc, oc_part_thread)


                            out = get_stats_fast(conv_params, tiling, part, order_type, verbose=False)

                            if out is None:
                                continue

                            stats, stationary = out
                            cycles = stats.total_cycles
                            energy = stats.energy

                            if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy > energy):

                                best_energy = energy
                                best_cycles = cycles
                                best_order = order_type
                                best_tiling = tiling
                                best_partitioning = part
                                best_dataflow = stationary


    return (best_tiling, best_order, best_cycles, best_energy, best_partitioning, best_dataflow)
