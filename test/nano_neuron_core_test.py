############################################
# Test: Nano Neuron Core
# Author: Mohit Upadhyay
############################################

import math
from random import seed
from random import random

max_bw = 256
kernel_size = 3
num_filters_max = 2

# Testing the Nano Neuron Core design
from ..arch.nano_neuron_core import Nano_neuron_core
from ..arch.activation_router import Activation_router

# Instantiate Nano Neuron Core
cycles = 0

nano_neuron_core_inst = Nano_neuron_core()
activation_router = [Activation_router() for i in range(max_bw)]

stride = 1
padding = 1

# Values to load into the Nano Neuron Core
weight_in_cpu = []
load_weight_sram = 1

# Seed  random number generators
seed(1)

for filter_num in range(num_filters_max):
	weight_in_cpu.append([])
	for row in range(int(kernel_size)):
		weight_in_cpu[filter_num].append([])
		for column in range(int(kernel_size)):
			weight_in_cpu[filter_num][row].append(random())

# Load weight from CPU/IO to SRAM bank in nano core
nano_neuron_core_inst.load_weight_from_cpu(weight_in_cpu, load_weight_sram)

# Input data to the Nano Neuron Core
data_in_val = []
load_weight_sram = 1
mac_en = 1

for row in range(max_bw):
	data_in_val.append(random())

# Start the Nano Core operation
for filter in range(len(weight_in_cpu)):
	nano_neuron_core_inst.conv2D(data_in_val, mac_en, stride, padding, filter)
	cycles = cycles + nano_neuron_core_inst.cycles

	#Print out the output values
	with open("react-sim/data/nano_neuron_core_output.txt", 'a+') as file:

		file.write("Nano neuron core output:\n")
		file.write("Output channel #" + str(filter) + "\n")
		for val in range(max_bw):
			file.write(str(nano_neuron_core_inst.data_out[val]))
			file.write(" ")

			if val == (max_bw - 1):
				file.write("\n")

with open("react-sim/data/nano_neuron_core_output.txt", 'a+') as file:
	file.write("Number of cycles = " + str(cycles))