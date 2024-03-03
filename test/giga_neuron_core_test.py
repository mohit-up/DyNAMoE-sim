############################################
# Test: Giga Neuron Core
# Author: Mohit Upadhyay
############################################

import numpy as np

max_bw = 256

# Testing the GIGA Neuron Core design
from ..arch.giga_neuron_core import Giga_neuron_core
from ..arch.activation_router_giga import Activation_router_giga
from ..arch.training_router import Training_router

# Instantiate GIGA Neuron Core

giga_neuron_core_inst = Giga_neuron_core()
activation_router_giga = [Activation_router_giga() for i in range(max_bw)]
training_router = [Training_router() for i in range(max_bw)]

# Values to load into the Nano Neuron Core
load_weight = 1
trans_en = 0
mac_en = 0

weight = np.random.rand(max_bw, max_bw)

# Load weight from CPU/IO to SRAM bank in nano core
giga_neuron_core_inst.load_weight_from_cpu(weight, load_weight)

# Input data to the Nano Neuron Core
load_weight = 0
trans_en = 0
mac_en = 1
inf_in_sel = 0

data_in_val = np.random.rand(1, max_bw)

# Start the GIGA Core operation
giga_neuron_core_inst.matrix_mul(data_in_val, mac_en, trans_en)

# Validate the output values
exp_data_out = np.matmul(data_in_val, weight)

np.savetxt("react-sim/data/giga_neuron_core_output.txt", giga_neuron_core_inst.data_out)