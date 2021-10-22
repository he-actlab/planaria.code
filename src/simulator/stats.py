class Stats(object):
    """
    Stores the stats from the simulator
    """

    def __init__(self):
        self.total_cycles = 0
        self.core_cycles = 0
        self.core_compuyte_cycles = 0
        self.mem_stall_cycles = 0
        self.energy = 0
        self.namespaces = ['act', 'wgt', 'out', 'dram', 'act_fifo', 'out_accum']
        self.reads = {}
        self.writes = {}
        for n in self.namespaces:
            self.reads[n] = 0
            self.writes[n] = 0
        self.data_dispatch_hops = 0
        self.wgt_bus_hops = 0




    def __iter__(self):
        return iter([\
                     self.total_cycles,
                     self.core_cycles,
                     self.core_compute_cycles,
                     self.mem_stall_cycles,
                     self.energy,
                     self.reads['act'],
                     self.reads['wgt'],
                     self.reads['out'],
                     self.reads['dram'],
                     self.reads['act_fifo'],
                     self.reads['out_acuum'],
                     self.writes['out'],
                     self.writes['dram'],
                    self.data_dispatch_hops,
                    self.wgt_bus_hops
                    ])

    def __add__(self, other):
        ret = Stats()
        ret.total_cycles = self.total_cycles + other.total_cycles
        ret.core_cycles = self.core_cycles + other.core_cycles
        ret.core_compute_cycles = self.core_compute_cycles + other.core_compute_cycles
        ret.energy = self.energy + other.energy
        ret.mem_stall_cycles = self.mem_stall_cycles + other.mem_stall_cycles
        ret.data_dispatch_hops = self.data_dispatch_hops + other.data_dispatch_hops
        ret.wgt_bus_hops = self.wgt_bus_hops + other.wgt_bus_hops
        for n in self.namespaces:
            ret.reads[n] = self.reads[n] + other.reads[n]
            ret.writes[n] = self.writes[n] + other.writes[n]
        return ret

    def __mul__(self, other):
        ret = Stats()
        ret.total_cycles = self.total_cycles * other
        ret.core_cycles = self.core_cycles * other
        ret.core_compute_cycles = self.core_compute_cycles * other
        ret.energy = self.energy * other
        ret.mem_stall_cycles = self.mem_stall_cycles * other
        ret.data_dispatch_hops = self.data_dispatch_hops * other
        ret.wgt_bus_hops = self.wgt_bus_hops * other
        for n in self.namespaces:
            ret.reads[n] = self.reads[n] * other
            ret.writes[n] = self.writes[n] * other
        return ret

    def __str__(self):
        ret = '\tStats'
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Total', self.total_cycles)
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Core', self.core_cycles)
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Core Compute', self.core_compute_cycles)
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Total', self.energy)
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Memory Stalls', self.mem_stall_cycles)
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Data Dispatch Hops', self.data_dispatch_hops)
        ret+= '\n\t{0:>20}   : {1:>20,}, '.format('Weight Bus Hops', self.wgt_bus_hops)
        ret+= '\n\tReads: '
        for n in self.namespaces:
            ret+= '\n\t{0:>20} rd: {1:>20,} bits, '.format(n, self.reads[n])
        ret+= '\n\tWrites: '
        for n in self.namespaces:
            ret+= '\n\t{0:>20} wr: {1:>20,} bits, '.format(n, self.writes[n])
        return ret

    def get_energy(self, energy_cost, dram_cost, interconnect_cost):
        dram_cost *= 1.e-3
        wgt_bus_cost = interconnect_cost['wgt_bus_cost'] * 1.e-3
        data_dispatch_cost = interconnect_cost['data_dispatch_cost'] * 1.e-3

        total_leak_energy, core_dyn_energy, wgt_sram_read_energy, wgt_sram_write_energy, act_sram_read_energy, act_sram_write_energy, out_sram_read_energy, out_sram_write_energy, act_fifo_read_energy, act_fifo_write_energy, out_accum_read_energy, out_accum_write_energy = energy_cost
        dyn_energy = (self.core_compute_cycles) * core_dyn_energy

        dyn_energy += self.reads['wgt'] * wgt_sram_read_energy
        dyn_energy += self.writes['wgt'] * wgt_sram_write_energy

        dyn_energy += self.reads['act'] * act_sram_read_energy
        dyn_energy += self.writes['act'] * act_sram_write_energy

        dyn_energy += self.reads['out'] * out_sram_read_energy
        dyn_energy += self.writes['out'] * out_sram_write_energy

        dyn_energy += self.reads['act_fifo'] * act_fifo_read_energy
        dyn_energy += self.writes['act_fifo'] * act_fifo_write_energy

        dyn_energy += self.reads['out_accum'] * out_accum_read_energy
        dyn_energy += self.writes['out_accum'] * out_accum_write_energy

        dyn_energy += self.reads['dram'] * dram_cost
        dyn_energy += self.writes['dram'] * dram_cost

        dyn_energy += self.data_dispatch_hops * data_dispatch_cost
        dyn_energy += self.wgt_bus_hops * wgt_bus_cost
        # Leakage Energy
        leak_energy = 0
        return dyn_energy + leak_energy

    def get_energy_breakdown(self, energy_cost, dram_cost, interconnect_cost):
        dram_cost *= 1.e-3
        wgt_bus_cost = interconnect_cost['wgt_bus_cost'] * 1.e-3
        data_dispatch_cost = interconnect_cost['data_dispatch_cost'] * 1.e-3
        total_leak_energy, core_dyn_energy, wgt_sram_read_energy, wgt_sram_write_energy, act_sram_read_energy, act_sram_write_energy, out_sram_read_energy, out_sram_write_energy, act_fifo_read_energy, act_fifo_write_energy, out_accum_read_energy, out_accum_write_energy = energy_cost
        core_energy = (self.core_compute_cycles) * core_dyn_energy
        breakdown = [core_energy]

        sram_energy = self.reads['wgt'] * wgt_sram_read_energy
        sram_energy += self.writes['wgt'] * wgt_sram_write_energy

        sram_energy += self.reads['act'] * act_sram_read_energy
        sram_energy += self.writes['act'] * act_sram_write_energy

        sram_energy += self.reads['out'] * out_sram_read_energy
        sram_energy += self.writes['out'] * out_sram_write_energy

        breakdown.append(sram_energy)

        fifo_accum_energy = self.reads['act_fifo'] * act_fifo_read_energy
        fifo_accum_energy += self.writes['act_fifo'] * act_fifo_write_energy

        fifo_accum_energy += self.reads['out_accum'] * out_accum_read_energy
        fifo_accum_energy += self.writes['out_accum'] * out_accum_write_energy

        breakdown.append(fifo_accum_energy)

        dram_energy = self.reads['dram'] * dram_cost
        dram_energy += self.writes['dram'] * dram_cost

        breakdown.append(dram_energy)

        interconnect_energy = self.data_dispatch_hops * data_dispatch_cost
        interconnect_energy += self.wgt_bus_hops * wgt_bus_cost

        breakdown.append(interconnect_energy)

        return breakdown
