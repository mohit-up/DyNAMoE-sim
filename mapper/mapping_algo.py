############################################
# Mapping Algorithm
# Author: Mohit Upadhyay
############################################

import argparse
import math
import sched
import sys
import numpy as np

from giga_core_scheduler import schedule_giga
from nano_core_scheduler import schedule_nano

max_dim = 256
max_img_dim = 16
num_nano_cores_y = 2
num_nano_cores_x = 4
num_giga_cores = 2
max_output_channels = 64

if (__name__ == '__main__'):

    # Get the arguments of the layer
    parser = argparse.ArgumentParser()
    parser.add_argument('-layer', metavar='Layer Type', type=str, default="FC", help="Path to the topology file")
    parser.add_argument('-input', metavar='Input Size', type=str, default="256", help="Input Size for FC / Img x dim in Conv")
    parser.add_argument('-output', metavar='Output Size', type=str, default="256", help="Output Size for FC / Img y dim in Conv")
    parser.add_argument('-kernel', metavar='Kernel Size', type=str, default="3", help="Kernel Size for the Conv. operation")
    parser.add_argument('-output_ch', metavar='Output Channels', type=str, default="1", help="Output Channels")
    parser.add_argument('-input_ch', metavar='Input Channels', type=str, default="1", help="Input Channels")

    # Get the configuration of the NN layer type
    args = parser.parse_args()
    layer_type = args.layer

    ####### FC Layer  scheduling ########
    if (layer_type == "FC"):

        # FC layer parameters
        input_size = int(args.input)
        output_size = int(args.output)

        # Partition the layer based on the number of GIGA cores
        row_div = math.ceil(input_size/max_dim)
        column_div = math.ceil(output_size/max_dim)

        row_dim = int(input_size/row_div)
        column_dim = int(output_size/column_div)

        num_giga_partitions = row_div * column_div
        num_sch = math.ceil(num_giga_partitions/num_giga_cores)

        weights = np.ones((input_size, output_size))
        weights_partition = []

        sch = 0

        if (column_div < num_giga_cores):
            for i in range(column_div):
                weights_partition.append([])
                for j in range(row_div):
                    temp = weights[(j * row_dim) : (j * row_dim + row_dim), (i * column_dim) : (i * column_dim + column_dim)]
                    weights_partition[i].append(temp)

            for i in range (column_div):
                for j in range(row_div):
                    if (j % 2 == 0):
                        print("Scheduling No. = " + str(sch + 1))
                        sch = sch + 1

                    row_start = j * row_dim
                    row_end = row_start + row_dim
                    column_start = i * column_dim
                    column_end = column_start + column_dim

                    schedule_giga((j % 2), row_start, row_end, column_start, column_end)

        else:
            for i in range(column_div):
                weights_partition.append([])
                for j in range(row_div):
                    temp = weights[(j * row_dim) : (j * row_dim + row_dim), (i * column_dim) : (i * column_dim + column_dim)]
                    weights_partition[i].append(temp)

            for i in range(column_div):
                for j in range(row_div):

                    if (j % 2 == 0):
                        print("Scheduling No. = " + str(sch + 1))
                        sch = sch + 1

                    row_start = j * row_dim
                    row_end = row_start + row_dim
                    column_start = i * column_dim
                    column_end = column_start + column_dim

                    schedule_giga((j % 2), row_start, row_end, column_start, column_end)

    ######### Conv. Layer scheduling #########
    else:

        output_channels = int(args.output_ch)
        input_channels = int(args.input_ch)
        kernel_size = int(args.kernel)
        img_x_dim = int(args.input)
        img_y_dim = int(args.output)

        assert kernel_size <= 7, "Max. supported kernel size is 7 x 7 (Only square kernels supported for now. 1D Conv to be added later)"

        if (img_x_dim < max_img_dim):
            img_x_range = int(img_x_dim)
        else:
            num_part_x = math.ceil(img_x_dim/max_img_dim)
            img_x_range = int(img_x_dim/num_part_x)

        if (img_y_dim < max_img_dim):
            img_y_range = int(img_y_dim)
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

        in_ch_sch = 1

        # Scheduling if image fits within REACT
        if (num_part_x < 2 and num_part_y < 2):
            num_in_ch_sch = (num_nano_cores_x * num_nano_cores_y)/output_ch_sch
            print("Num of schedulings = " + str(num_inst))
            
            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    if (i * num_nano_cores_y + j < output_ch_sch):
                        print("Input channel number = " + str(in_ch_sch))
                        
                        start_out_ch = (i * num_nano_cores_y + j) * num_output_channels_sch
                        end_out_ch = (i * num_nano_cores_y + j) * num_output_channels_sch + num_output_channels_sch
                        print("Output channel num = " + str(start_out_ch) + " - " + str(end_out_ch))
                    else:
                        if (in_ch_sch < num_in_ch_sch):
                            in_ch_sch = in_ch_sch + 1
                        print("Input channel number = " + str(in_ch_sch))
                        
                        start_out_ch = ((i - in_ch_sch) * num_nano_cores_y + j) * num_output_channels_sch
                        end_out_ch = ((i - in_ch_sch) * num_nano_cores_y + j) * num_output_channels_sch + num_output_channels_sch
                        print("Output channel num = " + str(start_out_ch) + " - " + str(end_out_ch))
                    img_x_start = 0
                    img_x_end = img_x_range
                    img_y_start = 0
                    img_y_end = img_y_range
                    # print("----- Scheduling No. = " + str(inst) + " -----")
                    schedule_nano(i, j, img_x_start, img_x_end, img_y_start, img_y_end)

        else:
            inst = 0
            in_ch = 0

            while (in_ch < input_channels):
                print("\n##### Input Channel no. = " + str(in_ch) + " #####\n")
                # Output channel scheduling
                sch = 0
                while (sch < output_ch_sch):
                    out_ch = 0
                    while (out_ch < num_output_channels_sch):
                        print("~~~~~~~ Output channel no. = " + str(sch * max_output_channels + out_ch) + " ~~~~~~~")
                        # Image scheduling along x
                        x = 0
                        while (x < num_inst_sch_x):
                            i = 0
                            while (i < inst_x):
                                # Image scheduling along y
                                y = 0
                                while (y < num_inst_sch_y):
                                    j = 0
                                    while (j < inst_y):
                                        img_x_start = (x * inst_x + i) * img_x_range
                                        img_x_end = img_x_start + img_x_range
                                        img_y_start = (y * inst_y + j) * img_y_range
                                        img_y_end = img_y_start + img_y_range
                                        schedule_nano(i, j, img_x_start, img_x_end, img_y_start, img_y_end)

                                        print("\n------ Scheduling no. = " + str(inst) + " -----\n")

                                        if (i == (num_nano_cores_x - 1) and j == (num_nano_cores_y - 1)):
                                            inst = inst + 1

                                        j = j + 1
                                    y = y + 1
                                i = i + 1
                            x = x + 1
                        out_ch = out_ch + 1
                    sch = sch + 1
                in_ch = in_ch + 1