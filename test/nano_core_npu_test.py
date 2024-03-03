############################################
# Test: Nano Core NPU
# Author: Mohit Upadhyay
############################################

max_bw = 256
filter_num = 64
kernel_size = 3

# Testing the PS router
from ..arch.nano_core_npu import Nano_core_npu

# Clock cycles are measured
cycles = 0

#Instantiate the activation routers and relu units
nano_core_npu_inst = Nano_core_npu()
sample_val = 1

# Weights to load into the Nano Neuron Core
weight_in_cpu = []
load_weight_sram = 1

for filter_num in range(filter_num):
	weight_in_cpu.append([])
	for row in range(int(kernel_size)):
		weight_in_cpu[filter_num].append([])
		for column in range(int(kernel_size)):
			weight_in_cpu[filter_num][row].append(sample_val)

# Load weight from CPU
nano_core_npu_inst.nano_neuron_core.load_weight_from_cpu(weight_in_cpu, load_weight_sram)

# Data & Control input to the Nano NPU Core
data_in_val = []
load_weight_sram = 0
mac_en = 1
stride = 1
padding = 1
inf_in_sel = 0

# PS NoC Control inputs
add_bypass_en = 0
sum_en = 0
ps_en = 1
consec_add_en = 0
add_in_sel = 2
add_out_sel = 4

# ReLU Control inputs
sum_or_local = 0
relu_en = 1

# Activation NoC Control inputs
inject_en = 1
ws_bypass_en = 0
act_in_sel = 2
act_out_sel = 3
axon_buffer_en = 0

for row in range(max_bw):
	data_in_val.append(sample_val)

with open("react-sim/data/nano-core-npu-output.txt", 'w') as file:
	file.write("")

# Forward Propagation (Neural Network inference)
for filter in range(len(nano_core_npu_inst.nano_neuron_core.weight_sram)):
	nano_core_npu_inst.nano_neuron_core.conv2D(data_in_val, mac_en, stride, padding, inf_in_sel, nano_core_npu_inst.activation_routers, filter)

	for router in range(max_bw):
		nano_core_npu_inst.ps_routers[router].set_in_port(data_in_val[router], data_in_val[router], data_in_val[router], data_in_val[router])
		nano_core_npu_inst.ps_routers[router].route_in(add_in_sel, add_bypass_en, sum_en, consec_add_en, nano_core_npu_inst.nano_neuron_core, router, filter)
		nano_core_npu_inst.ps_routers[router].route_out(add_out_sel, add_bypass_en, sum_en, ps_en, filter)

	# Relu operation in NPU
	nano_core_npu_inst.forward_pass_activation(sum_or_local, relu_en)
	
	for router in range(max_bw):
		nano_core_npu_inst.activation_routers[router].set_in_port(data_in_val[router], data_in_val[router], data_in_val[router], data_in_val[router])
		nano_core_npu_inst.activation_routers[router].route_in(act_in_sel, inject_en, nano_core_npu_inst.relu_units[router], filter)
		nano_core_npu_inst.activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, act_out_sel, filter)

	# Print the different output ports
	with open("react-sim/data/nano-core-npu-output.txt", 'a+') as file:

		file.write("\nNano neuron core output:\n")
		for router in range(max_bw):
			file.write(str(nano_core_npu_inst.nano_neuron_core.data_out[router]))
			file.write(" ")

		file.write("\n")

		file.write("PS NoC output:\n")
		for router in range(max_bw):
			file.write(str(nano_core_npu_inst.ps_routers[router].local_out))
			file.write(" ")

		file.write("\n")

		file.write("Nano Core NPU/Activation output:\n")
		for router in range(max_bw):
			file.write(str(nano_core_npu_inst.activation_routers[router].local_out))
			file.write(" ")

# Total number of cycles
cycles  = nano_core_npu_inst.nano_neuron_core.cycles + nano_core_npu_inst.ps_routers[router].cycles + nano_core_npu_inst.activation_routers[router].cycles

# Print the different output ports
with open("react-sim/data/nano-core-npu-output.txt", 'a+') as file:

	file.write("\n")
	file.write("\nNeuron core clock cycles = " )
	file.write(str(nano_core_npu_inst.nano_neuron_core.cycles))

	file.write("\n")
	file.write("\nPS router clock cycles = " )
	file.write(str(nano_core_npu_inst.ps_routers[router].cycles))

	file.write("\n")
	file.write("\nact router clock cycles = " )
	file.write(str(nano_core_npu_inst.activation_routers[router].cycles))

	file.write("\n")
	file.write("\nNumber of clock cycles = " )
	file.write(str(cycles))