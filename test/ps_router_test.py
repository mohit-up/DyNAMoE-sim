############################################
# Test: PS Router
# Author: Mohit Upadhyay
############################################

max_bw = 256
kernel_size = 3
num_filters_max = 64

# Testing the PS router
from ..arch.ps_router import PS_router
from ..arch.nano_neuron_core import Nano_neuron_core

ps_routers = []

# Instantiate the Nano neuron core
nano_neuron_core = Nano_neuron_core()

#Instantiate the activation routers and relu units
for router in range(max_bw):
	ps_routers.append(PS_router())

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

# Print output port values
with open("react-sim/outputs/ps_router_test.txt", 'w') as file:

	file.write("Neuron core output:\n")
	for router in range(max_bw):
		file.write(str(ps_routers[router].add_in_1))
		file.write(" ")

	if router == (max_bw - 1):
		file.write("\n")

	file.write("Local output:\n")
	for router in range(max_bw):
		file.write(str(ps_routers[router].local_out))
		file.write(" ")

	if router == (max_bw - 1):
		file.write("\n")

	file.write("North Output:\n")
	for router in range(max_bw):
		file.write(str(ps_routers[router].north_out))
		file.write(" ")

	if router == (max_bw - 1):
		file.write("\n")

	file.write("South Output:\n")
	for router in range(max_bw):
		file.write(str(ps_routers[router].south_out))
		file.write(" ")

	if router == (max_bw - 1):
		file.write("\n")

	file.write("East output:\n")
	for router in range(max_bw):
		file.write(str(ps_routers[router].east_out))
		file.write(" ")

	if router == (max_bw - 1):
		file.write("\n")

	file.write("West output:\n")
	for router in range(max_bw):
		file.write(str(ps_routers[router].west_out))
		file.write(" ")

	if router == (max_bw - 1):
		file.write("\n")