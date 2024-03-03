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
weights_layer_1 = np.ones((784, 512))
weights_layer_2 = np.ones((512, 10))

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

# Data input
input_data = np.ones((1, 784))
input_data_slice_1 = input_data[:, 0:196]
input_data_slice_2 = input_data[:, 196:392]
input_data_slice_3 = input_data[:, 392:588]
input_data_slice_4 = input_data[:, 588:784]
input_padding = np.zeros((1, 60))

input_data_padded = []
for i in range(4):
    input_data_padded.append(np.reshape(np.column_stack((input_data[:, i * 196:(i + 1) * 196], input_padding)), (256,)).tolist())

# Instantiate react chip
react_inst = React()

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
layer_2_input = []
# Control inputs for neuron cores
load_weight = 0
mac_en = 1
trans_en = 0
inf_in_sel = 1

# Start inference operation
for i in range(num_giga_cores):
    layer_2_input.append([])
    for j in range(1):
        react_inst.giga_cores[i][j].forward_pass(inf_in_sel, dummy, mac_en, trans_en)
        layer_2_input[i].append(react_inst.giga_cores[i][j].giga_neuron_core.data_in)

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

output = np.zeros((1, output_size))
one_hot = np.zeros((1, output_size))

for index in range(output_size):
    output[0][index] = react_inst.giga_cores[0][0].activation_routers[router].local_out

# Calculate the error gradient of outputs (done in CPU)
padding_array = np.zeros((1, 246))

output_softmax = np.exp(output)
total = np.sum(output_softmax)
output_softmax = output_softmax/total

# Ex:- Label value is 2
one_hot[0][2] = 1

input_grad = (output_softmax - one_hot)
input_grad = np.column_stack((input_grad, padding_array))
input_grad = np.reshape(input_grad, (input_grad.shape[1],)).tolist()

with open("react-sim/data/mnist-mlp-training-output.txt", "w") as file:
    file.write("Input grad \n")
    for router in range(len(input_data)):
        file.write(str(input_grad))
        file.write(" ")
    file.write("\n")

# Backpropagation
# Control inputs for neuron core
load_weight = 0
mac_en = 0
trans_en = 1
tr_in_sel = 0

# PS routers control inputs
add_input_sel = 2
bypass_en = 0
sum_en = 0
consec_add_en = 0
ps_en = 1
sum_en = 0
consec_add_en = 0
sel_tr_cb = 1

# Training router control inputs
train_input_sel = 0
train_inject_en = 1
train_axon_buffer_en = 1
train_bypass_en = 0
train_output_sel = 4

# Learning rate
learning_rate = 0.01

# Layer 2 scheduling
for i in range(num_giga_cores):
    for j in range(1):
        react_inst.giga_cores[i][j].backward_pass(input_grad, layer_2_input[i][j], mac_en, trans_en, tr_in_sel)

        # PS operation
        for router in range(max_bw):
            react_inst.giga_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, react_inst.giga_cores[i][j].giga_neuron_core, router)
            react_inst.giga_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

        # Training router operation
        for router in range(max_bw):
            react_inst.giga_cores[i][j].training_routers[router].route_in(train_input_sel, train_inject_en, react_inst.giga_cores[i][j].ps_routers[router])
            react_inst.giga_cores[i][j].training_routers[router].route_out(train_bypass_en, train_inject_en, train_axon_buffer_en, train_output_sel)

        # Weight update operation for Layer 2
        react_inst.giga_cores[i][j].weight_update_engine.weight_update_operation(weights_layer_2_list[i][j], learning_rate)

# Send updated weights to the neuron core
load_weight = 1
mac_en = 0
trans_en = 0

for i in range(num_giga_cores):
    for j in range(1):
        # Send updated weights
        react_inst.giga_cores[i][j].weight_update_engine.weight_output(react_inst.giga_cores[i][j].giga_neuron_core, load_weight)

with open("react-sim/data/mnist-mlp-training-output.txt", "a+") as file:
    for i in range(num_giga_cores):
        for j in range(1):
            file.write(str(react_inst.giga_cores[i][j].giga_neuron_core.weight))
            file.write(" ")
        file.write("\n")

# Layer 1 scheduling
for sch in range(3, -1, -1):

    # Control inputs for neuron core
    load_weight = 1
    mac_en =  0
    trans_en = 0

    for i in range(num_giga_cores):
        for j in range(1):
            react_inst.giga_cores[i][j].giga_neuron_core.load_weight_from_cpu(weights_layer_1_list[j+sch][i], load_weight)

    # Control inputs to the neuron core
    load_weight = 0
    mac_en = 0
    trans_en = 1
    tr_in_sel = 1

    # Control inputs to the training router
    train_input_sel = 4
    train_output_sel = 0
    train_inject_en = 0
    train_bypass_en = 1
    train_axon_buffer_en = 0

    # Control inputs to PS routers
    add_input_sel = 0
    add_output_sel = 1
    bypass_en = 0
    sum_en = 1
    consec_add_en = 0

    for i in range(num_giga_cores):
        for j in range(1):
            react_inst.giga_cores[i][j].backward_pass(dummy, input_data_padded[sch], mac_en, trans_en, tr_in_sel)

            # PS operation
            for router in range(max_bw):
                react_inst.giga_cores[i][j].ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, react_inst.giga_cores[i][j].giga_neuron_core, router)
                react_inst.giga_cores[i][j].ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

            # Training router operation
            for router in range(max_bw):
                react_inst.giga_cores[i][j].training_routers[router].route_in(train_input_sel, train_inject_en, react_inst.giga_cores[i][j].ps_routers[router])
                react_inst.giga_cores[i][j].training_routers[router].route_out(train_bypass_en, train_inject_en, train_axon_buffer_en, train_output_sel)

            # Weight update operation for Layer 2
            react_inst.giga_cores[i][j].weight_update_engine.weight_update_operation(weights_layer_1_list[j+sch][i], learning_rate)

    # Control inputs for loading weights
    load_weight = 1
    mac_en = 0
    trans_en = 0

    for i in range(num_giga_cores):
        for j in range(1):
            react_inst.giga_cores[i][j].weight_update_engine.weight_output(react_inst.giga_cores[i][j].giga_neuron_core, load_weight)

    with open("react-sim/data/mnist-mlp-training-output.txt", "a+") as file:
        for i in range(num_giga_cores):
            for j in range(1):
                file.write(str(react_inst.giga_cores[i][j].giga_neuron_core.weight))
                file.write(" ")
            file.write("\n")