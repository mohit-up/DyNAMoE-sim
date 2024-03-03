############################################
# MobileNetv1 benchmark on REACT chip (inference)
# Author: Mohit Upadhyay
############################################

max_bw = 256
num_nano_cores_y = 2
num_nano_cores_x = 4
num_giga_cores = 2

# NN config
output_size = 10

import numpy as np

from ..arch.react import React

# Numer of clock cycles
cycles = 0
dummy = np.zeros((256,)).tolist()

# Convolutional Layer 1 weights
conv_weights_layer_1 = np.ones((32, 3, 3))

# Segregate conv1 based on input channels

# Data input (entire test set)
# input_data = np.loadtxt("react-sim/data/mnist_test_input.txt", delimiter=",")
input_data = np.ones((1, 3, 28, 28))

# Instantiate react chip
react_inst = React()

# Iterate through each test set
for sample in range(len(input_data)):

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    for i in range(num_nano_cores_x):
        for j in range(num_nano_cores_y):
            react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_1, load_weight)

    # Inference operation
    # Neuron core control inputs
    load_weight = 0
    mac_en = 1
    trans_en = 0
    inf_in_sel = 0

    stride = 2
    padding = 1
    
    ##### CONV LAYER 1 ######
    for filter in range(len(conv_weights_layer_1)):

        # Set the input channel number
        channel = 1

        # Pad the input data (required for the calculation of PS outputs at the edges)
        input_data_slice_1 = np.pad(input_data[sample:sample+1, channel-1:channel, 0:14, 0:14], 1, mode = 'constant')
        input_data_slice_2 = np.pad(input_data[sample:sample+1, channel-1:channel, 0:14, 14:28], 1, mode = 'constant')
        input_data_slice_3 = np.pad(input_data[sample:sample+1, channel-1:channel, 14:28, 0:14], 1, mode = 'constant')
        input_data_slice_4 = np.pad(input_data[sample:sample+1, channel-1:channel, 14:28, 14:28], 1, mode = 'constant')

        input_data_slice_1 = np.reshape(input_data_slice_1, (max_bw, ))
        input_data_slice_2 = np.reshape(input_data_slice_2, (max_bw, ))
        input_data_slice_3 = np.reshape(input_data_slice_3, (max_bw, ))
        input_data_slice_4 = np.reshape(input_data_slice_4, (max_bw, ))

        input_data_padded = []
        input_data_padded.extend((input_data_slice_1.tolist(), input_data_slice_2.tolist(), input_data_slice_3.tolist(), input_data_slice_4.tolist()))

        # Schedule 1st input channel for Layer 1
        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                react_inst.nano_cores[i][j].activation_routers, filter)

        # PS operation (only for edges)
        # Send PS data from cores (n, 0) -> (n, 1)
        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                # PS router control inputs
                if ((j % 2) == 0):
                    add_input_sel = 2
                    add_output_sel = 5
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 1
                    ps_en = 0

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

                else:
                    add_input_sel = 0
                    add_output_sel = 3
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 0
                    ps_en = 1

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)

        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                # PS router control inputs
                if ((j % 2) == 0):
                    add_input_sel = 0
                    add_output_sel = 2
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 0
                    ps_en = 1

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)

                else:
                    add_input_sel = 3
                    add_output_sel = 0
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 1
                    ps_en = 0

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Send PS data from cores (0, n) -> (1, n)
        for i in range(num_nano_cores_x/2):
            # PS router control inputs
            if ((i % 2) == 0):
                add_input_sel = 4
                add_output_sel = 1
                consec_add_en = 0
                bypass_en = 0
                sum_en = 0
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

            else:
                add_input_sel = 1
                add_output_sel = 5
                consec_add_en = 0
                bypass_en = 0
                sum_en = 1
                ps_en = 0

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

        for i in range(num_nano_cores_x/2):
            # PS router control inputs
            if ((i % 2) == 0):
                add_input_sel = 4
                add_output_sel = 0
                consec_add_en = 0
                bypass_en = 0
                sum_en = 1
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

            else:
                add_input_sel = 1
                add_output_sel = 5
                consec_add_en = 0
                bypass_en = 0
                sum_en = 0
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
        
        # Perform routing
        react_inst.perform_routing()

        # PS operation for the corner pixels
        # Core (1, 1) -> Core (0, 1)
        add_input_sel = 3
        add_output_sel = 3
        bypass_en = 0
        sum_en = 0
        ps_en = 1
        consec_add_en = 0

        react_inst.nano_cores[1][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, ps_en,
        react_inst.nano_cores[1][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[1][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Core (0, 1) -> Core (0, 0)
        add_input_sel = 1
        add_output_sel = 2
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[0][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[0][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # PS operation at Core (0, 0)
        add_input_sel = 1
        add_output_sel = 2
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[0][0].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[0][0].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][0].ps_routers[255].gen_sum(sum_en)
        react_inst.nano_cores[0][0].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Core (0, 0) -> Core (0, 1)
        add_input_sel = 3
        add_output_sel = 2
        bypass_en = 0
        sum_en = 0
        ps_en = 1
        consec_add_en = 0

        react_inst.nano_cores[0][0].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, ps_en,
        react_inst.nano_cores[0][0].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][0].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # Core (0, 1) -> Core (1, 1)
        add_input_sel = 3
        add_output_sel = 1
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[0][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[0][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # PS operation at Core (1, 1)
        add_input_sel = 0
        add_output_sel = 2
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[1][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[1][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[1][1].ps_routers[255].gen_sum(sum_en)
        react_inst.nano_cores[1][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Store output of the computed partial sum for first channel in the PS buffer
        add_input_sel = 0
        add_output_sel = 5
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0

        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[i][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # Schedule for Input channel 2
        # Set the input channel number
        channel = 2

        # Pad the input data (required for the calculation of PS outputs at the edges)
        input_data_slice_1 = np.pad(input_data[sample:sample+1, channel-1:channel, 0:14, 0:14], 1, mode = 'constant')
        input_data_slice_2 = np.pad(input_data[sample:sample+1, channel-1:channel, 0:14, 14:28], 1, mode = 'constant')
        input_data_slice_3 = np.pad(input_data[sample:sample+1, channel-1:channel, 14:28, 0:14], 1, mode = 'constant')
        input_data_slice_4 = np.pad(input_data[sample:sample+1, channel-1:channel, 14:28, 14:28], 1, mode = 'constant')

        input_data_slice_1 = np.reshape(input_data_slice_1, (max_bw, ))
        input_data_slice_2 = np.reshape(input_data_slice_2, (max_bw, ))
        input_data_slice_3 = np.reshape(input_data_slice_3, (max_bw, ))
        input_data_slice_4 = np.reshape(input_data_slice_4, (max_bw, ))

        input_data_padded = []
        input_data_padded.extend((input_data_slice_1.tolist(), input_data_slice_2.tolist(), input_data_slice_3.tolist(), input_data_slice_4.tolist()))

        for i in range(num_nano_cores_x/2, num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[(i - 2) * 2 + (j - 2)], mac_en, stride, padding, inf_in_sel,
                react_inst.nano_cores[i][j].activation_routers, filter)

        # Send PS data from cores (n, 0) -> (n, 1)
        for i in range(num_nano_cores_x/2, num_nano_cores_x):
            for j in range(num_nano_cores_y):
                # PS router control inputs
                if ((j % 2) == 0):
                    add_input_sel = 2
                    add_output_sel = 5
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 1
                    ps_en = 0

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

                else:
                    add_input_sel = 0
                    add_output_sel = 3
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 0
                    ps_en = 1

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)

        for i in range(num_nano_cores_x/2, num_nano_cores_x):
            for j in range(num_nano_cores_y):
                # PS router control inputs
                if ((j % 2) == 0):
                    add_input_sel = 0
                    add_output_sel = 2
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 0
                    ps_en = 1

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)

                else:
                    add_input_sel = 3
                    add_output_sel = 0
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 1
                    ps_en = 0

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Send PS data from cores (0, n) -> (1, n)
        for i in range(num_nano_cores_x/2, num_nano_cores_x):
            # PS router control inputs
            if ((i % 2) == 0):
                add_input_sel = 4
                add_output_sel = 1
                consec_add_en = 0
                bypass_en = 0
                sum_en = 0
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

            else:
                add_input_sel = 1
                add_output_sel = 5
                consec_add_en = 0
                bypass_en = 0
                sum_en = 1
                ps_en = 0

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

        for i in range(num_nano_cores_x/2, num_nano_cores_x):
            # PS router control inputs
            if ((i % 2) == 0):
                add_input_sel = 4
                add_output_sel = 0
                consec_add_en = 0
                bypass_en = 0
                sum_en = 1
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

            else:
                add_input_sel = 1
                add_output_sel = 5
                consec_add_en = 0
                bypass_en = 0
                sum_en = 0
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
        
        # Perform routing
        react_inst.perform_routing()

        # PS operation for the corner pixels
        # Core (3, 3) -> Core (2, 3)
        add_input_sel = 3
        add_output_sel = 3
        bypass_en = 0
        sum_en = 0
        ps_en = 1
        consec_add_en = 0
        react_inst.nano_cores[3][3].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, ps_en,
        react_inst.nano_cores[3][3].nano_neuron_core, 255, filter)
        react_inst.nano_cores[3][3].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # Core (2, 3) -> Core (2, 2)
        add_input_sel = 1
        add_output_sel = 2
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0
        react_inst.nano_cores[2][3].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[2][3].nano_neuron_core, 255, filter)
        react_inst.nano_cores[2][3].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # PS operation at Core (2, 2)
        add_input_sel = 1
        add_output_sel = 2
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0
        react_inst.nano_cores[2][2].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[2][2].nano_neuron_core, 255, filter)
        react_inst.nano_cores[2][2].ps_routers[255].gen_sum(sum_en)
        react_inst.nano_cores[2][2].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Core (2, 2) -> Core (2, 3)
        add_input_sel = 3
        add_output_sel = 2
        bypass_en = 0
        sum_en = 0
        ps_en = 1
        consec_add_en = 0
        react_inst.nano_cores[2][2].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, ps_en,
        react_inst.nano_cores[2][2].nano_neuron_core, 255, filter)
        react_inst.nano_cores[2][2].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # Core (2, 3) -> Core (3, 3)
        add_input_sel = 3
        add_output_sel = 1
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0
        react_inst.nano_cores[2][3].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[2][3].nano_neuron_core, 255, filter)
        react_inst.nano_cores[2][3].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # PS operation at Core (3, 3)
        add_input_sel = 0
        add_output_sel = 2
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0
        react_inst.nano_cores[3][3].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[3][3].nano_neuron_core, 255, filter)
        react_inst.nano_cores[3][3].ps_routers[255].gen_sum(sum_en)
        react_inst.nano_cores[3][3].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Send output of the computed partial sum for 2nd channel for addition to 1st channel
        add_input_sel = 0
        add_output_sel = 0
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0

        for i in range(num_nano_cores_x/2, num_nano_cores_x):
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[i][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        add_input_sel = 0
        add_output_sel = 0
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0

        for i in range((num_nano_cores_x/2 - 1), (num_nano_cores_x - 1)):
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[i][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        add_input_sel = 0
        add_output_sel = 5
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0

        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[i][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

    # Load weights for 3rd channel
    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    for i in range(num_nano_cores_x):
        for j in range(num_nano_cores_y):
            react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_1, load_weight)

    for filter in range(len(conv_weights_layer_1)):

        # Schedule 3rd input channel for Layer 1
        # Set the input channel number
        channel = 3

        # Pad the input data (required for the calculation of PS outputs at the edges)
        input_data_slice_1 = np.pad(input_data[sample:sample+1, channel-1:channel, 0:14, 0:14], 1, mode = 'constant')
        input_data_slice_2 = np.pad(input_data[sample:sample+1, channel-1:channel, 0:14, 14:28], 1, mode = 'constant')
        input_data_slice_3 = np.pad(input_data[sample:sample+1, channel-1:channel, 14:28, 0:14], 1, mode = 'constant')
        input_data_slice_4 = np.pad(input_data[sample:sample+1, channel-1:channel, 14:28, 14:28], 1, mode = 'constant')

        input_data_slice_1 = np.reshape(input_data_slice_1, (max_bw, ))
        input_data_slice_2 = np.reshape(input_data_slice_2, (max_bw, ))
        input_data_slice_3 = np.reshape(input_data_slice_3, (max_bw, ))
        input_data_slice_4 = np.reshape(input_data_slice_4, (max_bw, ))

        input_data_padded = []
        input_data_padded.extend((input_data_slice_1.tolist(), input_data_slice_2.tolist(), input_data_slice_3.tolist(), input_data_slice_4.tolist()))

        # Control inputs
        load_weight = 0
        mac_en = 1
        trans_en = 0

        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                react_inst.nano_cores[i][j].activation_routers, filter)

        # PS operation (only for edges)
        # Send PS data from cores (n, 0) -> (n, 1)
        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                # PS router control inputs
                if ((j % 2) == 0):
                    add_input_sel = 2
                    add_output_sel = 5
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 1
                    ps_en = 0

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

                else:
                    add_input_sel = 0
                    add_output_sel = 3
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 0
                    ps_en = 1

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)

        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                # PS router control inputs
                if ((j % 2) == 0):
                    add_input_sel = 0
                    add_output_sel = 2
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 0
                    ps_en = 1

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)

                else:
                    add_input_sel = 3
                    add_output_sel = 0
                    consec_add_en = 0
                    bypass_en = 0
                    sum_en = 1
                    ps_en = 0

                    for row in range(16):
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
                        react_inst.nano_cores[i][j].ps_routers[row * 16 + 14].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (row * 16 + 14), filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Send PS data from cores (0, n) -> (1, n)
        for i in range(num_nano_cores_x/2):
            # PS router control inputs
            if ((i % 2) == 0):
                add_input_sel = 4
                add_output_sel = 1
                consec_add_en = 0
                bypass_en = 0
                sum_en = 0
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

            else:
                add_input_sel = 1
                add_output_sel = 5
                consec_add_en = 0
                bypass_en = 0
                sum_en = 1
                ps_en = 0

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

        for i in range(num_nano_cores_x/2):
            # PS router control inputs
            if ((i % 2) == 0):
                add_input_sel = 4
                add_output_sel = 0
                consec_add_en = 0
                bypass_en = 0
                sum_en = 1
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].gen_sum(sum_en)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)

            else:
                add_input_sel = 1
                add_output_sel = 5
                consec_add_en = 0
                bypass_en = 0
                sum_en = 0
                ps_en = 1

                for j in range(num_nano_cores_y):
                    for column in range(16):
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
                        react_inst.nano_cores[i][j].ps_routers[15 * 16 + column].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i][j].nano_neuron_core, (15 * 16 + column), filter)
        
        # Perform routing
        react_inst.perform_routing()

        # PS operation for the corner pixels
        # Core (1, 1) -> Core (0, 1)
        add_input_sel = 3
        add_output_sel = 3
        bypass_en = 0
        sum_en = 0
        ps_en = 1
        consec_add_en = 0

        react_inst.nano_cores[1][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, ps_en,
        react_inst.nano_cores[1][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[1][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # Core (0, 1) -> Core (0, 0)
        add_input_sel = 1
        add_output_sel = 2
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[0][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[0][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # PS operation at Core (0, 0)
        add_input_sel = 1
        add_output_sel = 2
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[0][0].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[0][0].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][0].ps_routers[255].gen_sum(sum_en)
        react_inst.nano_cores[0][0].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Core (0, 0) -> Core (0, 1)
        add_input_sel = 3
        add_output_sel = 2
        bypass_en = 0
        sum_en = 0
        ps_en = 1
        consec_add_en = 0

        react_inst.nano_cores[0][0].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, ps_en,
        react_inst.nano_cores[0][0].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][0].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # Core (0, 1) -> Core (1, 1)
        add_input_sel = 3
        add_output_sel = 1
        bypass_en = 1
        sum_en = 0
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[0][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[0][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[0][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # PS operation at Core (1, 1)
        add_input_sel = 0
        add_output_sel = 2
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0

        react_inst.nano_cores[1][1].ps_routers[255].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
        react_inst.nano_cores[1][1].nano_neuron_core, 255, filter)
        react_inst.nano_cores[1][1].ps_routers[255].gen_sum(sum_en)
        react_inst.nano_cores[1][1].ps_routers[255].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Add the computer partial sum for channel 3 to the previous partial sum val in PS buffer
        add_input_sel = 5
        add_output_sel = 4
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        consec_add_en = 0

        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[i][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # Final computed sum sent to the Activation NoC
        # Activation control inputs
        inject_en = 1
        ws_bypass_en = 0
        axon_buffer_en = 0
        act_input_sel = 2
        act_output_sel = 4

        # Relu Unit
        relu_en = 1
        sum_or_local = 1

        # Relu Unit & Activation Routers' operation
        for i in range(num_nano_cores_x/2):
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].forward_pass_activation(relu_en, sum_or_local)
                    react_inst.nano_cores[i][j].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                    react_inst.nano_cores[i][j].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)

        # Concatenate the outputs of layer 1

        # Outputs of Core (0, 1) -> Core (0, 0)
        # Activation control inputs
        inject_en = 0
        ws_bypass_en = 1
        axon_buffer_en = 0
        act_input_sel = 4
        act_output_sel = 3

        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[0][1].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[0][1].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)

        # Activation control inputs
        inject_en = 0
        ws_bypass_en = 1
        axon_buffer_en = 0
        act_input_sel = 2
        act_output_sel = 4

        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Outputs of Core (1, 0) -> Core (0, 0)
        # Activation control inputs
        inject_en = 0
        ws_bypass_en = 1
        axon_buffer_en = 0
        act_input_sel = 4
        act_output_sel = 0

        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[1][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[1][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)

        # Activation control inputs
        inject_en = 0
        ws_bypass_en = 1
        axon_buffer_en = 0
        act_input_sel = 1
        act_output_sel = 4

        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)
        
        # Perform routing
        react_inst.perform_routing()

        # Outputs of Core (1, 1) -> Core (0, 0)
        # Activation control inputs
        inject_en = 0
        ws_bypass_en = 1
        axon_buffer_en = 0
        act_input_sel = 4
        act_output_sel = 0

        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[1][1].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[1][1].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)

        # Activation control inputs
        inject_en = 0
        ws_bypass_en = 1
        axon_buffer_en = 0
        act_input_sel = 1
        act_output_sel = 3

        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[0][1].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[0][1].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)

        # Activation control inputs
        inject_en = 0
        ws_bypass_en = 1
        axon_buffer_en = 0
        act_input_sel = 2
        act_output_sel = 4

        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, filter)
        
        # Perform routing
        react_inst.perform_routing()

    ########## Conv Layer 2 scheduling (Depth-wise convolution)#############
    # Weights of the conv layer after splitting (kept the same for now: will be different for different layers)
    conv_weights_layer_2 = np.ones((4, 3, 3))

    # Divide weights into different cores
    conv_weights_layer_2_list = [conv_weights_layer_2, conv_weights_layer_2, conv_weights_layer_2, conv_weights_layer_2, conv_weights_layer_2, conv_weights_layer_2,
    conv_weights_layer_2, conv_weights_layer_2]

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv2 i/p channel scheduled to one core (Core (0, 0) - (4, 1))
    for i in range(num_nano_cores_x):
        for j in range(num_nano_cores_y):
            react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_2_list[i * num_nano_cores_y + j], load_weight)

    # Neuron Core Control inputs
    load_weight = 0
    mac_en = 1
    trans_en = 0
    inf_in_sel = 1
    stride = 1
    padding = 1

    # Convolution operation
    for sch in range(4):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                react_inst.nano_cores[i][j].activation_routers, sch)

    # Activation control inputs
    inject_en = 1
    ws_bypass_en = 0
    axon_buffer_en = 1
    act_input_sel = 2
    act_output_sel = 4

    # Relu Unit
    relu_en = 1
    sum_or_local = 1

    for sch in range(len(conv_weights_layer_2)):
        # Outputs sent to the Activation NoC directly
        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[0][0].forward_pass_activation(relu_en, sum_or_local)
            react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, sch)
        
        # Perform routing
        react_inst.perform_routing()

    ########## Conv Layer 3 scheduling (Point-wise convolution) #############
    # Weights of the conv layer
    conv_weights_layer_3 = np.ones((64, 1, 1))

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv3 i/p channel scheduled to a separate core (Core (0, 0) - (4, 1))
    for sch in range(len(conv_weights_layer_2)):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_3, load_weight)

        # Neuron Core Control inputs
        load_weight = 0
        mac_en = 1
        trans_en = 0
        inf_in_sel = 1
        stride = 1
        padding = 0

        # Convolution operation
        for filter in range(len(conv_weights_layer_3)):
            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                    react_inst.nano_cores[i][j].activation_routers, filter)
    
            # Send PS from cores (3, 0) - (3, 1) -> (2, 0) - (2, 1) and (1, 0) - (1, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 0
            add_output_sel = 0
            bypass_en = 0
            sum_en = 0
            ps_en = 1
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2 + 1][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].gen_sum(sum_en)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
            # Perform routing
            react_inst.perform_routing()

            # Send PS from cores (2, 0) - (2, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[2][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[1][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][j].ps_routers[router].gen_sum(sum_en)
                    react_inst.nano_cores[0][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
            # Perform routing
            react_inst.perform_routing()

            # Send final PS from core (0, 1) -> core (0, 0)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 3
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][1].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][1].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][1].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 5
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
            # Perform routing
            react_inst.perform_routing()

        # For each output filter, send final sum to the activation NoC buffer
        for output_filter in range(len(conv_weights_layer_3)):
        
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 4
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            for router in range(max_bw):
                react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Activation control inputs
            inject_en = 1
            ws_bypass_en = 0
            axon_buffer_en = 0
            act_input_sel = 0
            act_output_sel = 4

            # Relu control inputs
            relu_en = 1
            sum_or_local = 1

            # Activation router operation
            for router in range(max_bw):
                react_inst.nano_cores[0][0].forward_pass_activation(relu_en, sum_or_local)
                react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, output_filter)

    ########### FINISHED TILL HERE $$$$$$$$$$$$$$$$

    ########## Conv Layer 4 scheduling (Depth-wise convolution)#############
    # Weights of the conv layer after splitting (kept the same for now: will be different for different layers)
    conv_weights_layer_4 = np.ones((8, 3, 3))

    # Divide weights into different cores
    conv_weights_layer_4_list = [conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4,
    conv_weights_layer_4, conv_weights_layer_4]

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv4 i/p channel scheduled to one core (Core (0, 0) - (4, 1))
    for i in range(num_nano_cores_x):
        for j in range(num_nano_cores_y):
            react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_4_list[i * num_nano_cores_y + j], load_weight)

    # Neuron Core Control inputs
    load_weight = 0
    mac_en = 1
    trans_en = 0
    inf_in_sel = 1
    stride = 2
    padding = 1

    # Convolution operation
    for sch in range(len(conv_weights_layer_4)):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                react_inst.nano_cores[i][j].activation_routers, sch)

    # Activation control inputs
    inject_en = 1
    ws_bypass_en = 0
    axon_buffer_en = 1
    act_input_sel = 2
    act_output_sel = 4

    # Relu Unit
    relu_en = 1
    sum_or_local = 1

    for sch in range(len(conv_weights_layer_4)):
        # Outputs sent to the Activation NoC directly
        # Activation router operation
        for router in range(max_bw):
            react_inst.nano_cores[0][0].forward_pass_activation(relu_en, sum_or_local)
            react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
            react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, sch)
        
        # Perform routing
        react_inst.perform_routing()

    ########## Conv Layer 5 scheduling (Point-wise convolution) #############
    # Weights of the conv layer
    conv_weights_layer_5 = np.ones((128, 1, 1))

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv5 i/p channel scheduled to a separate core (Core (0, 0) - (4, 1))
    for sch in range(len(conv_weights_layer_4)):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_5, load_weight)

        # Neuron Core Control inputs
        load_weight = 0
        mac_en = 1
        trans_en = 0
        inf_in_sel = 1
        stride = 1
        padding = 0

        # Convolution operation
        for filter in range(len(conv_weights_layer_5)):
            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                    react_inst.nano_cores[i][j].activation_routers, filter)
    
            # Send PS from cores (3, 0) - (3, 1) -> (2, 0) - (2, 1) and (1, 0) - (1, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 0
            add_output_sel = 0
            bypass_en = 0
            sum_en = 0
            ps_en = 1
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2 + 1][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].gen_sum(sum_en)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
            # Perform routing
            react_inst.perform_routing()

            # Send PS from cores (2, 0) - (2, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[2][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[1][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][j].ps_routers[router].gen_sum(sum_en)
                    react_inst.nano_cores[0][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
            # Perform routing
            react_inst.perform_routing()

            # Send final PS from core (0, 1) -> core (0, 0)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 3
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][1].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][1].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][1].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 5
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)
        
            # Perform routing
            react_inst.perform_routing()

        # For each output filter, send final sum to the activation NoC buffer
        for output_filter in range(len(conv_weights_layer_5)):
        
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 4
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0
            
            for router in range(max_bw):
                react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Activation control inputs
            inject_en = 1
            ws_bypass_en = 0
            axon_buffer_en = 0
            act_input_sel = 0
            act_output_sel = 4

            # Relu control inputs
            relu_en = 1
            sum_or_local = 1

            # Activation router operation
            for router in range(max_bw):
                react_inst.nano_cores[0][0].forward_pass_activation(relu_en, sum_or_local)
                react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, output_filter)

    ########## Conv Layer 6 scheduling (Depth-wise convolution)#############
    # Weights of the conv layer after splitting (kept the same for now: will be different for different layers)
    conv_weights_layer_6 = np.ones((16, 3, 3))

    # Divide weights into different cores
    conv_weights_layer_6_list = [conv_weights_layer_6, conv_weights_layer_6, conv_weights_layer_6, conv_weights_layer_6, conv_weights_layer_6, conv_weights_layer_6,
    conv_weights_layer_6, conv_weights_layer_6]

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv4 i/p channel scheduled to one core (Core (0, 0) - (4, 1))
    for i in range(num_nano_cores_x):
        for j in range(num_nano_cores_y):
            react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_6_list[i * num_nano_cores_y + j], load_weight)

    # Neuron Core Control inputs
    load_weight = 0
    mac_en = 1
    trans_en = 0
    inf_in_sel = 1
    stride = 1
    padding = 1

    # Convolution operation
    for sch in range(len(conv_weights_layer_6)):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                react_inst.nano_cores[i][j].activation_routers, sch)

    # Activation control inputs
    inject_en = 1
    ws_bypass_en = 0
    axon_buffer_en = 1
    act_input_sel = 2
    act_output_sel = 4

    # Relu Unit
    relu_en = 1
    sum_or_local = 1

    for sch in range(len(conv_weights_layer_6)):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                # Outputs sent to the Activation NoC directly
                # Activation router operation
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].forward_pass_activation(relu_en, sum_or_local)
                    react_inst.nano_cores[i][j].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                    react_inst.nano_cores[i][j].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, sch)

    ########## Conv Layer 7 scheduling (Point-wise convolution) #############
    # Weights of the conv layer
    conv_weights_layer_7 = np.ones((128, 1, 1))

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv5 i/p channel scheduled to a separate core (Core (0, 0) - (4, 1))
    for sch in range(len(conv_weights_layer_6)):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_5, load_weight)

        # Neuron Core Control inputs
        load_weight = 0
        mac_en = 1
        trans_en = 0
        inf_in_sel = 1
        stride = 1
        padding = 0

        # Convolution operation
        for filter in range(len(conv_weights_layer_6)):
            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                    react_inst.nano_cores[i][j].activation_routers, filter)
    
            # Send PS from cores (3, 0) - (3, 1) -> (2, 0) - (2, 1) and (1, 0) - (1, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 0
            add_output_sel = 0
            bypass_en = 0
            sum_en = 0
            ps_en = 1
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2 + 1][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].gen_sum(sum_en)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Send PS from cores (2, 0) - (2, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[2][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[1][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][j].ps_routers[router].gen_sum(sum_en)
                    react_inst.nano_cores[0][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Send final PS from core (0, 1) -> core (0, 0)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 3
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][1].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][1].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][1].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 5
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # For each output filter, send final sum to the activation NoC buffer
        for output_filter in range(len(conv_weights_layer_5)):
        
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 4
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                        react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Activation control inputs
            inject_en = 1
            ws_bypass_en = 0
            axon_buffer_en = 0
            act_input_sel = 0
            act_output_sel = 4

            # Relu control inputs
            relu_en = 1
            sum_or_local = 1

            # Activation router operation
            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i][j].forward_pass_activation(relu_en, sum_or_local)
                        react_inst.nano_cores[i][j].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                        react_inst.nano_cores[i][j].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, output_filter)

    ########## Conv Layer 4 scheduling (Depth-wise convolution)#############
    # Weights of the conv layer after splitting (kept the same for now: will be different for different layers)
    conv_weights_layer_4 = np.ones((8, 3, 3))

    # Divide weights into different cores
    conv_weights_layer_4_list = [conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4, conv_weights_layer_4,
    conv_weights_layer_4, conv_weights_layer_4]

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv4 i/p channel scheduled to one core (Core (0, 0) - (4, 1))
    for i in range(num_nano_cores_x):
        for j in range(num_nano_cores_y):
            react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_4_list[i * num_nano_cores_y + j], load_weight)

    # Neuron Core Control inputs
    load_weight = 0
    mac_en = 1
    trans_en = 0
    inf_in_sel = 1
    stride = 2
    padding = 1

    # Convolution operation
    for sch in range(8):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                react_inst.nano_cores[i][j].activation_routers, sch)

    # Activation control inputs
    inject_en = 1
    ws_bypass_en = 0
    axon_buffer_en = 1
    act_input_sel = 2
    act_output_sel = 4

    # Relu Unit
    relu_en = 1
    sum_or_local = 1

    for sch in range(8):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                # Outputs sent to the Activation NoC directly
                # Activation router operation
                for router in range(max_bw):
                    react_inst.nano_cores[i][j].forward_pass_activation(relu_en, sum_or_local)
                    react_inst.nano_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                    react_inst.nano_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, sch)

    ########## Conv Layer 5 scheduling (Point-wise convolution) #############
    # Weights of the conv layer
    conv_weights_layer_5 = np.ones((128, 1, 1))

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    # Each Conv5 i/p channel scheduled to a separate core (Core (0, 0) - (4, 1))
    for sch in range(4):
        for i in range(num_nano_cores_x):
            for j in range(num_nano_cores_y):
                react_inst.nano_cores[i][j].nano_neuron_core.load_weight_from_cpu(conv_weights_layer_5, load_weight)

        # Neuron Core Control inputs
        load_weight = 0
        mac_en = 1
        trans_en = 0
        inf_in_sel = 1
        stride = 1
        padding = 0

        # Convolution operation
        for filter in range(32):
            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    react_inst.nano_cores[i][j].nano_neuron_core.conv2D(input_data_padded[i * 2 + j], mac_en, stride, padding, inf_in_sel,
                    react_inst.nano_cores[i][j].activation_routers, filter)
    
            # Send PS from cores (3, 0) - (3, 1) -> (2, 0) - (2, 1) and (1, 0) - (1, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 0
            add_output_sel = 0
            bypass_en = 0
            sum_en = 0
            ps_en = 1
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2 + 1][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2 + 1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for i in range(num_nano_cores_x/2):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[i * 2][j].nano_neuron_core, router, filter)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].gen_sum(sum_en)
                        react_inst.nano_cores[i * 2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Send PS from cores (2, 0) - (2, 1) -> (0, 0) - (0, 1)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[2][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[2][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[2][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 0
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[1][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[1][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[1][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 1
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][j].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][j].ps_routers[router].gen_sum(sum_en)
                    react_inst.nano_cores[0][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Send final PS from core (0, 1) -> core (0, 0)
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 3
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][1].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][1].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][1].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # PS control inputs
            add_input_sel = 5
            add_output_sel = 5
            bypass_en = 0
            sum_en = 1
            ps_en = 0
            consec_add_en = 0

            # PS router operation
            for j in range(num_nano_cores_y):
                for router in range(max_bw):
                    react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                    react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                    react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

        # For each output filter, send final sum to the activation NoC buffer
        for output_filter in range(len(conv_weights_layer_5)):
        
            # PS control inputs
            add_input_sel = 5
            add_output_sel = 4
            bypass_en = 1
            sum_en = 0
            ps_en = 0
            consec_add_en = 0

            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en,
                        react_inst.nano_cores[0][0].nano_neuron_core, router, filter)
                        react_inst.nano_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, filter)

            # Activation control inputs
            inject_en = 1
            ws_bypass_en = 0
            axon_buffer_en = 0
            act_input_sel = 0
            act_output_sel = 4

            # Relu control inputs
            relu_en = 1
            sum_or_local = 1

            # Activation router operation
            for i in range(num_nano_cores_x):
                for j in range(num_nano_cores_y):
                    for router in range(max_bw):
                        react_inst.nano_cores[i][j].forward_pass_activation(relu_en, sum_or_local)
                        react_inst.nano_cores[i][j].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                        react_inst.nano_cores[i][j].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel, output_filter)

    with open("react-sim/data/mobilenet-v1-output.txt", "a+") as file:
        for router in range(output_size):
            file.write(str(react_inst.giga_cores[0][0].activation_routers[router].local_out))
            file.write(" ")
        file.write("\n")
        # file.write("Clock cycles = ")
        # file.write(str(cycles))