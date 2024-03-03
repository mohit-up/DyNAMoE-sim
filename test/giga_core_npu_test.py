############################################
# Test: GIGA Core NPU
# Author: Mohit Upadhyay
############################################

import numpy as np
max_bw = 256

# Testing the Activation router + ReLU unit
from ..arch.giga_core_npu import Giga_core_npu
sample_val = 1

# Total number of clock cycles measured
cycles = 0

# Weight Inputs to the NPU
weight_in_val = np.ones((256, 256))
weight_in_val = weight_in_val.tolist()

# Data Inputs to the NPU
data_input_val = np.ones((256, ))
data_input_val = data_input_val.tolist()

# Control inputs
# Neuron Core control inputs
mac_en = 0
trans_en = 0
load_weight = 1

# PS control inputs
add_input_sel = 2
add_output_sel = 4
bypass_en = 0
sum_en = 0
ps_en = 1
add_en = 0
consec_add_en = 0
sel_tr_cb = 0

# ReLU Unit
sum_or_local = 1
relu_en = 1

# Activation control inputs
inject_en = 1
ws_bypass_en = 0
axon_buffer_en = 0
act_input_sel = 2
act_output_sel = 3

# Instantiate the GIGA Nano NPU
giga_core_npu_inst = Giga_core_npu()

# Load weight into the GIGA core
giga_core_npu_inst.giga_neuron_core.load_weight_from_cpu(weight_in_val, load_weight)

# Control inputs to the neuron core
load_weight = 0
mac_en = 1
inf_in_sel = 0

# Start inference operation in the GIGA core
giga_core_npu_inst.forward_pass(inf_in_sel, data_input_val, mac_en, trans_en)

cycles = cycles + giga_core_npu_inst.giga_neuron_core.cycles

# PS operation
for router in range(max_bw):
	giga_core_npu_inst.ps_routers[router].set_in_port(data_input_val[router], data_input_val[router], data_input_val[router], data_input_val[router])
	giga_core_npu_inst.ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, giga_core_npu_inst.giga_neuron_core, router)
	giga_core_npu_inst.ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

cycles = cycles + giga_core_npu_inst.ps_routers[router].cycles

# Relu Unit & Activation Routers' inputs
for router in range(max_bw):
	giga_core_npu_inst.forward_pass_activation(relu_en, sum_or_local)
	giga_core_npu_inst.activation_routers[router].set_in_port(data_input_val[router], data_input_val[router], data_input_val[router], data_input_val[router])
	giga_core_npu_inst.activation_routers[router].route_in(act_input_sel, inject_en, giga_core_npu_inst.relu_units[router])
	giga_core_npu_inst.activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_output_sel)

cycles = cycles + giga_core_npu_inst.activation_routers[router].cycles

# Print the output port values in the GIGA core NPU
with open("react-sim/data/giga_core_npu_output.txt", 'w') as file:
	file.write("Inference operation \n")

	file.write("Neuron Core output: \n")
	for router in range(max_bw):
		file.write(str(giga_core_npu_inst.giga_neuron_core.data_out[router]))
		file.write(" ")

	file.write("\nLocal output of the PS NoC: \n")
	for router in range(max_bw):
		file.write(str(giga_core_npu_inst.ps_routers[router].local_out))
		file.write(" ")

	file.write("\nLocal output of the activation NoC: \n")
	for router in range(max_bw):
		file.write(str(giga_core_npu_inst.activation_routers[router].local_out))
		file.write(" ")

	file.write("\nClock cycles consumed for inference = ")
	file.write(str(cycles))

cycles = 0

# Control inputs for training
load_weight = 0
mac_en = 0
trans_en = 1
sel_tr_cb = 1
tr_in_sel = 0

# Start training operation in the GIGA core & 1D systolic array
giga_core_npu_inst.backward_pass(data_input_val, data_input_val, mac_en, trans_en, tr_in_sel)

# PS operation
for router in range(max_bw):
	giga_core_npu_inst.ps_routers[router].set_in_port(data_input_val[router], data_input_val[router], data_input_val[router], data_input_val[router])
	giga_core_npu_inst.ps_routers[router].route_in(add_input_sel, bypass_en, sum_en, consec_add_en, giga_core_npu_inst.giga_neuron_core, router)
	giga_core_npu_inst.ps_routers[router].route_out(add_output_sel, bypass_en, sum_en, ps_en, sel_tr_cb)

cycles = cycles + giga_core_npu_inst.ps_routers[router].cycles

# Control inputs for training routers
train_input_sel = 0
train_output_sel = 2
train_bypass_en = 0
train_inject_en = 1
train_axon_buffer_en = 1

# Training Router operation
for router in range(max_bw):
	giga_core_npu_inst.training_routers[router].set_in_port(data_input_val[router], data_input_val[router], data_input_val[router], data_input_val[router])
	giga_core_npu_inst.training_routers[router].route_in(train_input_sel, train_inject_en, giga_core_npu_inst.ps_routers[router])
	giga_core_npu_inst.training_routers[router].route_out(train_bypass_en, train_inject_en, train_axon_buffer_en, train_output_sel)

cycles = cycles + giga_core_npu_inst.training_routers[router].cycles

# Weight Update engine operation
learning_rate = 0.01
giga_core_npu_inst.weight_update_engine.weight_update_operation(weight_in_val, learning_rate)

cycles = cycles + giga_core_npu_inst.weight_update_engine.cycles

# Send updated weights to the neuron core
load_weight = 1
mac_en = 0
trans_en = 0

giga_core_npu_inst.weight_update_engine.weight_output(giga_core_npu_inst.giga_neuron_core, load_weight)

cycles = cycles + giga_core_npu_inst.giga_neuron_core.cycles

# Print the output port values in the GIGA core NPU
with open("react-sim/data/giga_core_npu_output.txt", 'a+') as file:
	file.write("\n Training operation \n")
	
	file.write("\n Neuron Core output: \n")
	for router in range(max_bw):
		file.write(str(giga_core_npu_inst.giga_neuron_core.data_out[router]))
		file.write(" ")

	file.write("\n Weight gradient output: \n")
	file.write(str(giga_core_npu_inst.weight_update_engine.weight_grad_in))
	file.write(" ")

	file.write("\n Updated weight: \n")
	file.write(str(giga_core_npu_inst.weight_update_engine.weight_in))
	file.write(" ")

	file.write("\n Local input of the PS NoC: \n")
	for router in range(max_bw):
		file.write(str(giga_core_npu_inst.ps_routers[router].train_add_in))
		file.write(" ")

	file.write("\n Local input of the training NoC: \n")
	for router in range(max_bw):
		file.write(str(giga_core_npu_inst.training_routers[router].local_in))
		file.write(" ")

	file.write("\n Output of the training NoC: \n")
	for router in range(max_bw):
		file.write(str(giga_core_npu_inst.training_routers[router].east_out))
		file.write(" ")

	file.write("\nClock cycles consumed for training (forward and backward pass) = ")
	file.write(str(cycles))