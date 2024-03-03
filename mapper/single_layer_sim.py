############################################
# Generate the different logical partitions for a single layer
# To generate the training and test data
# Author: Mohit Upadhyay
############################################

import argparse
import math
import sched
import sys
import numpy as np

from utils.topology_utils import topologies as topo
from mapper.giga_core_scheduler import schedule_giga
from arch.react import React

# Core architectural features
max_dim = 256
max_img_dim = 16
max_output_channels = 64

# Config is fixed for now
# Num of different types of cores (Currently config is fixed)
num_nano_cores_y = 0
num_nano_cores_x = 0
# num_giga_cores_x = 4
# num_giga_cores_y = 3

class single_layer_sim():

    def __init__(self):
        self.layer_name = None
        self.layer_id = 0
        self.layer_type = None
        self.load_cycles = 0
        self.cycles = 0
        
        # GEMM/FC layer
        self.batch_size = 0
        self.input_size = 0
        self.output_size = 0
        
        # Conv layer
        self.img_x_dim = 0
        self.img_y_dim = 0
        self.kernel_size = 0
        self.input_channels = 0
        self.output_channels = 0

    def reset_cycles(self):
        self.load_cycles = 0
        self.cycles = 0

    def set_params_single_layer(self, layer_id, entries):
        self.layer_id = layer_id
        self.layer_name = entries[layer_id][0]
        self.layer_type = entries[layer_id][1]
        
        # GEMM/FC layer
        self.batch_size = entries[layer_id][2]
        self.input_size = entries[layer_id][3]
        self.output_size = entries[layer_id][7]

        # Conv layer
        self.img_x_dim = entries[layer_id][2]
        self.img_y_dim = entries[layer_id][3]
        self.kernel_size = entries[layer_id][4]
        self.input_channels = entries[layer_id][6]
        self.output_channels = entries[layer_id][7]

    def get_layer_id(self):
        return self.layer_id
        
    def run_single_layer(self, num_giga_cores_x, num_giga_cores_y):
        ####### FC Layer  scheduling ########
        if (self.layer_type == "FC"):

            # FC layer parameters
            input_size = int(self.input_size)
            output_size = int(self.output_size)

            # Partition the layer based on the number of GIGA cores
            row_div = math.ceil(input_size/max_dim)
            column_div = math.ceil(output_size/max_dim)

            row_dim = int(input_size/row_div)
            column_dim = int(output_size/column_div)

            num_giga_partitions = row_div * column_div
            num_sch = math.ceil(num_giga_partitions/(num_giga_cores_x * num_giga_cores_y))

            print("Logical GIGA partitions: ", str(num_giga_partitions))
            print("Num of schedulings: ", str(num_sch))

            # print("Input Neurons ", str(0), "from ", str(row_dim))
            # print("Output Neurons ", str(0), "from ", str(column_dim))

            print("Logical Mappings (x, y): (" + str(row_div) + str(", ") + str(column_div) + str(")"))

            # Instantiate the REACT instance (based on the num of GIGA cores)
            react_12g_inst = React(row_dim, column_dim, num_giga_cores_x, num_giga_cores_y)

            # Calculate number of clock cycles for running single layer
            total_cycles = 0

            for num in range(num_sch):
                # Generate the control signals according to the core mappings
                ps_in_sel, ps_out_sel, ws_in_sel, ws_out_sel, bypass_en, sum_en, ps_en, inject_en, ws_bypass_en = schedule_giga(input_size, output_size, row_div, column_div, num, num_giga_cores_x, num_giga_cores_y)

                # Load weights and perform inference
                react_12g_inst.load_weight_from_cpu(1)
                react_12g_inst.forward_pass(ps_in_sel, ps_out_sel, ws_in_sel, ws_out_sel, bypass_en, sum_en, ps_en, inject_en, ws_bypass_en)
                num_cycles, load_cycles = react_12g_inst.report_perf()
                
                total_cycles += num_cycles

            # Compute total number of clock cycles across different batches
            self.load_cycles = load_cycles * self.batch_size
            self.cycles = total_cycles * self.batch_size

        ######### Conv. Layer scheduling #########
        elif (self.layer_type == "Conv"):

            output_channels = int(self.output_channels)
            input_channels = int(self.input_channels)
            kernel_size = int(self.kernel_size)
            img_x_dim = int(self.input_size)
            img_y_dim = int(self.output_size)

            assert kernel_size <= 7, "Max. supported kernel size is 7 x 7 (Only square kernels supported for now. 1D Conv to be added later)"

            if (img_x_dim < max_img_dim):
                img_x_range = int(img_x_dim)
                num_part_x = int(math.ceil(img_x_dim/max_img_dim))
            else:
                num_part_x = math.ceil(img_x_dim/max_img_dim)
                img_x_range = int(img_x_dim/num_part_x)

            if (img_y_dim < max_img_dim):
                img_y_range = int(img_y_dim)
                num_part_y = int(math.ceil(img_y_dim/max_img_dim))
            else:
                num_part_y = math.ceil(img_y_dim/max_img_dim)
                img_y_range = int(img_y_dim/num_part_y)
                
            # print(num_part_x)
            # print(num_part_y)
            num_inst = math.ceil(num_part_x/num_nano_cores_x * num_part_y/num_nano_cores_y * output_channels/max_output_channels * input_channels)

            # Output channel partitioning
            output_ch_sch = math.ceil(output_channels/max_output_channels)
            if (output_channels % max_output_channels == 0):
                num_output_channels_sch = max_output_channels
            else:
                num_output_channels_sch = output_channels % max_output_channels

            # Image x partitioning
            num_inst_sch_x = math.ceil(num_part_x/num_nano_cores_x)
            if (num_part_x % num_nano_cores_x == 0):
                inst_x = num_nano_cores_x
            else:
                inst_x = num_part_x % num_nano_cores_x

            # Image y partitioning
            num_inst_sch_y = math.ceil(num_part_y/num_nano_cores_y)
            if (num_part_y % num_nano_cores_y == 0):
                inst_y = num_nano_cores_y
            else:
                inst_y = num_part_y % num_nano_cores_y

            # in_ch_sch = 1

            # Scheduling if image fits within REACT
            if (num_part_x < 2 and num_part_y < 2):
                num_in_ch_sch = (num_nano_cores_x * num_nano_cores_y)/output_ch_sch

        # Print the number of cycles
        print("\nClock Cycles (this layer): " + str(self.cycles))