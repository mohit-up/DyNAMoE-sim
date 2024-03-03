############################################
# Test: Activation Router
# Author: Mohit Upadhyay
############################################

max_bw = 256
kernel_size = 3
num_filters_max = 64

# Testing the Activation router + ReLU unit
from ..arch.activation_router import Activation_router
from ..arch.relu_unit import Relu_unit
from ..arch.nano_neuron_core import Nano_neuron_core
from ..arch.ps_router import PS_router

ps_routers = []
activation_routers = []
relu_units = []

# Instantiate the Nano neuron core
nano_neuron_core = Nano_neuron_core()

# Instantiate the PS routers and relu units
for router in range(max_bw):
	ps_routers.append(PS_router())

#Instantiate the activation routers and relu units
for router in range(max_bw):
	activation_routers.append(Activation_router())
	relu_units.append(Relu_unit())

# Neuron Core input and outputs
data_in = [1] * max_bw
weight_in_cpu = []

# Data Input
input_val = []
total_sum = []
local_ps = []

sample_val = 1

# Values to load into the Nano Neuron Core
weight_in_cpu = []
load_weight_sram = 1
mac_en = 0

for filter_num in range(num_filters_max):
	weight_in_cpu.append([])
	for row in range(int(kernel_size)):
		weight_in_cpu[filter_num].append([])
		for column in range(int(kernel_size)):
			weight_in_cpu[filter_num][row].append(sample_val)

# Values to be sent to the PS routers
for router in range(max_bw):
	input_val.append(sample_val)

# Load weights to Nano neuron core
nano_neuron_core.load_weight_from_cpu(weight_in_cpu, load_weight_sram)

# Nano core operation
load_weight_sram = 0
mac_en = 1
stride = 1
padding = 1
nano_neuron_core.conv2D(data_in, mac_en, stride, padding)	

# PS Control inputs
bypass_en = 0
sum_en = 0
ps_en = 1
consec_add_en = 0
input_sel = 2
output_sel = 4

# PS router operation
for router in range(max_bw):
	ps_routers[router].route_in(input_val[router], input_val[router], input_val[router], input_val[router], input_sel, bypass_en, sum_en, consec_add_en, nano_neuron_core, router)
	ps_routers[router].route_out(output_sel, bypass_en, sum_en, ps_en)

# Control inputs
relu_en = 1
inject_en = 1
ws_bypass_en = 0
axon_buffer_en = 0
input_sel = 2
output_sel = 3

for router in range(max_bw):
	relu_units[router].relu_comp(relu_en, ps_routers[router])

	activation_routers[router].route_in(input_val[router], input_val[router], input_val[router], input_val[router], input_sel, inject_en, relu_units[router])
	activation_routers[router].route_out(ws_bypass_en, inject_en, axon_buffer_en, output_sel)

# Print output port values
with open("react-sim/data/activation_router_output.txt", 'w') as file:

	file.write("Neuron Core output:\n")
	for router in range(max_bw):
		file.write(str(relu_units[router].input))
		file.write(" ")

		if router == (max_bw - 1):
			file.write("\n")

	file.write("Local output:\n")
	for router in range(max_bw):
		file.write(str(activation_routers[router].local_out))
		file.write(" ")

		if router == (max_bw - 1):
			file.write("\n")

	file.write("North output:\n")
	for router in range(max_bw):
		file.write(str(activation_routers[router].north_out))
		file.write(" ")

		if router == (max_bw - 1):
			file.write("\n")

	file.write("South output:\n")
	for router in range(max_bw):
		file.write(str(activation_routers[router].south_out))
		file.write(" ")

		if router == (max_bw - 1):
			file.write("\n")

	file.write("East output:\n")
	for router in range(max_bw):
		file.write(str(activation_routers[router].east_out))
		file.write(" ")

		if router == (max_bw - 1):
			file.write("\n")

	file.write("West output:\n")
	for router in range(max_bw):
		file.write(str(activation_routers[router].west_out))
		file.write(" ")

		if router == (max_bw - 1):
			file.write("\n")