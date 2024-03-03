############################################
# MNIST-MLP benchmark on REACT chip (inference)
# Author: Mohit Upadhyay
############################################

max_bw = 256
num_nano_cores_y = 2
num_nano_cores_x = 4
num_giga_cores = 2

# NN config
input_size = 784
output_size = 10

import numpy as np

from ..arch.react import React

# Numer of clock cycles
cycles = 0
dummy = np.zeros((256,)).tolist()

# Weights to be read into the cores
weights_layer_1 = np.loadtxt("react-sim/data/weights_layer_1.txt", delimiter=",")
# weights_layer_1 = np.ones((784, 512))
weights_layer_2 = np.loadtxt("react-sim/data/weights_layer_2.txt", delimiter=",")
# weights_layer_2 = np.ones((512, 10))

# Split the weights to be loaded
weights_layer_1_list = []
weights_layer_2_list = []

# Pad all the slices to make the matrix shape 256 x 256
padding_array = np.zeros((60, 256))
for i in range(4):
    weights_layer_1_list.append([])
    for j in range(2):
        weights_layer_1_list[i].append(weights_layer_1[0 + (i * 196):196 + (i * 196), 0 + (j * 256):256 + (j * 256)])
        weights_layer_1_list[i][j] = np.vstack((weights_layer_1_list[i][j], padding_array)).tolist()

# Pad all the slices to make the matrix shape 256 x 256
padding_array = np.zeros((256, 246))
for i in range(2):
    weights_layer_2_list.append([])
    for j in range(1):
        weights_layer_2_list[i].append(weights_layer_2[0 + (i * 256):256 + (i * 256),:])
        weights_layer_2_list[i][j] = np.column_stack((weights_layer_2_list[i][j], padding_array)).tolist()

# Data input (entire test set)
input_data = np.loadtxt("react-sim/data/mnist_test_input.txt", delimiter=",")

# Instantiate react chip
react_inst = React()

# Iterate through each test set
for sample in range(len(input_data)):
    input_data_slice_1 = input_data[:, 0:196]
    input_data_slice_2 = input_data[:, 196:392]
    input_data_slice_3 = input_data[:, 392:588]
    input_data_slice_4 = input_data[:, 588:784]
    input_padding = np.zeros((1, 60))

    input_data_padded = []
    for i in range(4):
        input_data_padded.append(np.reshape(np.column_stack((input_data[sample:sample+1, i * 196:(i + 1) * 196], input_padding)), (256,)).tolist())

    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    for i in range(num_giga_cores):
        for j in range(1):
            react_inst.giga_cores[i][j].giga_neuron_core.load_weight_from_cpu(weights_layer_1_list[j][i], load_weight)

    # Inference operation
    # Neuron core control inputs
    load_weight = 0
    mac_en = 1
    trans_en = 0
    inf_in_sel = 0

    # PS router control inputs
    add_input_sel = 0
    add_output_sel = 5
    consec_add_en = 0
    bypass_en = 0
    sum_en = 0
    ps_en = 1
    sel_tr_cb = 0

    # Start inference operation
    for i in range(num_giga_cores):
        for j in range(1):
            react_inst.giga_cores[i][j].forward_pass(inf_in_sel, input_data_padded[0], mac_en, trans_en)

            # PS operation
            for router in range(max_bw):
                react_inst.giga_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, react_inst.giga_cores[i][j].giga_neuron_core, router)
                react_inst.giga_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

    # Next three partitions for Layer 1
    for sch in range(3):
        # Control inputs
        load_weight = 1
        mac_en = 0
        trans_en = 0

        for i in range(num_giga_cores):
            for j in range(1):
                react_inst.giga_cores[i][j].giga_neuron_core.load_weight_from_cpu(weights_layer_1_list[j+(sch+1)][i], load_weight)

        # Inference operation
        # Neuron core control inputs
        load_weight = 0
        mac_en = 1
        trans_en = 0
        inf_in_sel = 0

        # PS router control inputs
        if (sch == 2):
            add_output_sel = 4
        else:
            add_output_sel = 5

        add_input_sel = 4
        consec_add_en = 0
        bypass_en = 0
        sum_en = 1
        ps_en = 0
        sel_tr_cb = 0

        # Start inference operation
        for i in range(num_giga_cores):
            for j in range(1):
                react_inst.giga_cores[i][j].forward_pass(inf_in_sel, input_data_padded[sch+1], mac_en, trans_en)

                # PS operation
                for router in range(max_bw):
                    react_inst.giga_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, react_inst.giga_cores[i][j].giga_neuron_core, router)
                    react_inst.giga_cores[i][j].ps_routers[router].gen_sum(sum_en)
                    react_inst.giga_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

    # Relu Unit
    relu_en = 1
    sum_or_local = 1

    # Activation control inputs
    inject_en = 1
    ws_bypass_en = 0
    axon_buffer_en = 0
    act_input_sel = 2
    act_output_sel = 4

    # Relu Unit & Activation Routers' inputs
    for i in range(num_giga_cores):
        for j in range(1):
            for router in range(max_bw):
                react_inst.giga_cores[i][j].forward_pass_activation(relu_en, sum_or_local)
                react_inst.giga_cores[i][j].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[i][j].relu_units[router])
                react_inst.giga_cores[i][j].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel)

    # Layer 2 scheduling
    # Control inputs
    load_weight = 1
    mac_en = 0
    trans_en = 0

    for i in range(num_giga_cores):
        for j in range(1):
            react_inst.giga_cores[i][j].giga_neuron_core.load_weight_from_cpu(weights_layer_2_list[i][j], load_weight)

    # Inference (layer 2)
    # Control inputs for neuron cores
    load_weight = 0
    mac_en = 1
    trans_en = 0
    inf_in_sel = 1

    # Start inference operation
    for i in range(num_giga_cores):
        for j in range(1):
            react_inst.giga_cores[i][j].forward_pass(inf_in_sel, dummy, mac_en, trans_en)

    # PS control inputs
    add_input_sel = 2
    add_output_sel = 0
    bypass_en = 0
    ps_en = 1
    sum_en = 0
    consec_add_en = 0
    sel_tr_cb = 0

    # PS operation
    for router in range(max_bw):
        react_inst.giga_cores[1][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, react_inst.giga_cores[1][0].giga_neuron_core, router)
        react_inst.giga_cores[1][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

    # PS control inputs
    add_input_sel = 1
    add_output_sel = 4
    bypass_en = 0
    ps_en = 0
    sum_en = 1
    consec_add_en = 0
    sel_tr_cb = 0

    # Routing operation
    react_inst.perform_routing()

    # PS operation
    for router in range(max_bw):
        react_inst.giga_cores[0][0].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, react_inst.giga_cores[0][0].giga_neuron_core, router)
        react_inst.giga_cores[0][0].ps_routers[router].gen_sum(sum_en)
        react_inst.giga_cores[0][0].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

    # Relu Unit
    relu_en = 0
    sum_or_local = 1

    # Activation control inputs
    inject_en = 1
    ws_bypass_en = 0
    axon_buffer_en = 0
    act_input_sel = 2
    act_output_sel = 4

    # Relu Unit & Activation Routers' inputs
    for router in range(max_bw):
        react_inst.giga_cores[0][0].forward_pass_activation(relu_en, sum_or_local)
        react_inst.giga_cores[0][0].activation_routers[router].route_in(act_input_sel, inject_en, react_inst.giga_cores[0][0].relu_units[router])
        react_inst.giga_cores[0][0].activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel)

    cycles = react_inst.giga_cores[0][0].giga_neuron_core.cycles + react_inst.giga_cores[0][0].ps_routers[router].cycles + react_inst.giga_cores[0][0].activation_routers[router].cycles

    with open("react-sim/data/mnist-mlp-output.txt", "a+") as file:
        #for router in range(output_size):
        #    file.write(str(react_inst.giga_cores[0][0].activation_routers[router].local_out))
        #    file.write(" ")
        file.write("\n")
        file.write("Clock cycles = ")
        file.write(str(cycles))
